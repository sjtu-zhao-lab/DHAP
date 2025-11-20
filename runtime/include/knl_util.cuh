#include <util.h>
#include "cuco/static_multimap.cuh"
#include "cuco/static_map.cuh"
#include "cuco/sentinel.cuh"
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>

using hash_value_type = int;
using size_type = uint32_t;
constexpr auto hash_sentinel = std::numeric_limits<hash_value_type>::max();
constexpr auto row_idx_sentinel = std::numeric_limits<size_type>::max();

using join_ht_type = cuco::static_multimap<hash_value_type, size_type>;
using join_ht_mview = cuco::static_multimap<hash_value_type, size_type>::device_mutable_view;
using join_ht_view = cuco::static_multimap<hash_value_type, size_type>::device_view;

struct groupby_ht_keyT {
  size_type idx[4];
  constexpr groupby_ht_keyT() noexcept : idx{row_idx_sentinel} {}
  __host__ __device__ __forceinline__
  groupby_ht_keyT(size_type i0, size_type i1 = row_idx_sentinel, 
                  size_type i2 = row_idx_sentinel, size_type i3 = row_idx_sentinel) {
    idx[0] = i0;
    idx[1] = i1;
    idx[2] = i2;
    idx[3] = i3;
  }

  __host__ __device__ __forceinline__
  constexpr bool operator==(const groupby_ht_keyT& other) {
    for (int i = 0; i < 4; i++) {
      if (idx[i] != other.idx[i]) 
        return false;
    }
    return true;
    // return (idx[0] == other.idx[0]) && (idx[1] == other.idx[1]) 
    //     && (idx[2] == other.idx[2]) && (idx[3] == other.idx[3]);
  }
};

constexpr groupby_ht_keyT groupby_ht_empty_key{};
using gb_ht_type = cuco::static_map<groupby_ht_keyT, size_type>;
using gb_ht_mview = gb_ht_type::device_mutable_view;
using gb_ht_view = gb_ht_type::device_view;
// constexpr groupby_ht_keyT groupby_ht_empty_key{row_idx_sentinel, row_idx_sentinel, row_idx_sentinel, row_idx_sentinel};

template<typename Iterator>
class batch_iterator : public thrust::iterator_adaptor<batch_iterator<Iterator>, Iterator>
{
public:
  typedef thrust::iterator_adaptor<batch_iterator<Iterator>, Iterator> super_t;

  __host__ __device__
  batch_iterator(const Iterator* batches, const row_size_t* batches_size, int num_batches)
    : super_t(*batches), begin_(batches[0]), batches_(batches), 
      batches_size_(batches_size), num_batches_(num_batches) {}

  friend class thrust::iterator_core_access;

private:
  const Iterator* batches_;
  const Iterator begin_;
  const row_size_t* batches_size_;
  const int num_batches_;
  __host__ __device__
  typename super_t::reference dereference() const {
    auto t = get_ptr_from_batches_generic(this->base()-begin_);
    return *t;
  }

  __device__ __forceinline__
  Iterator get_ptr_from_batches_generic(row_size_t global_idx) const {
    int batch_idx = 0;
    row_size_t idx = global_idx;
    for (int b = 0; b < num_batches_; b++) {
      if (idx >= batches_size_[b]) {
        idx -= batches_size_[b];
      }
      else {
        batch_idx = b;
        break;
      }
    }
    return batches_[batch_idx] + idx;
  }
};

struct get_first : public thrust::unary_function<groupby_ht_keyT, row_size_t>
{
  const int n_;
  get_first(int n) : n_(n) {}

  __host__ __device__
  row_size_t operator()(groupby_ht_keyT keys) const {
    return keys.idx[n_];
  }
};

__device__ __forceinline__
int get_from_batches(row_size_t global_idx, int** batches, 
                     row_size_t* batches_size, int num_batches)
{
  int batch_idx = 0;
  row_size_t idx = global_idx;
  for (int b = 0; b < num_batches; b++) {
    if (idx >= batches_size[b]) {
      idx -= batches_size[b];
    }
    else {
      batch_idx = b;
      break;
    }
  }
// if (batch_idx > 0)
//   printf("%d %d %d\n", batch_idx, global_idx, idx);
  return batches[batch_idx][idx];
}

__device__ __forceinline__
int* get_ptr_from_batches(row_size_t global_idx, int** batches, 
                          row_size_t* batches_size, int num_batches)
{
  int batch_idx = 0;
  row_size_t idx = global_idx;
  for (int b = 0; b < num_batches; b++) {
    if (idx >= batches_size[b]) {
      idx -= batches_size[b];
    }
    else {
      batch_idx = b;
      break;
    }
  }
  return batches[batch_idx] + idx;
}

class pair_equality
{
public:
  pair_equality(int* left_join_key_iter, int* right_join_key_iter)
    : _left_join_key_iter(left_join_key_iter), 
      _right_join_key_iter(right_join_key_iter)
  {}

  template <typename LhsPair, typename RhsPair>
  __device__ __forceinline__ 
  auto operator()(LhsPair const& lhs, RhsPair const& rhs) const noexcept
  {
    return lhs.first == rhs.first && 
           _left_join_key_iter[rhs.second] == _right_join_key_iter[lhs.second];
  }

private:
  int* _left_join_key_iter;
  int* _right_join_key_iter;
};

class batch_pair_equality
{
public:
  batch_pair_equality(int* left_join_key_iter,
                      // int* right_t, 
                      // int* right_t1, 
                      int** right_join_key_batch_iter,
                      row_size_t* right_batch_size,
                      int num_right_batches
                      )
    : _left_join_key_iter(left_join_key_iter), 
      _dev_right_join_key_batch_iter(right_join_key_batch_iter),
      _right_batch_size(right_batch_size),
      _num_right_batches(num_right_batches)
  {
    // cudaMalloc(&_dev_right_join_key_batch_iter, sizeof(int*)*num_right_batches);
    // cudaMemcpy(_dev_right_join_key_batch_iter, right_join_key_batch_iter,
    //            sizeof(int*)*num_right_batches, cudaMemcpyHostToDevice);
  }

  template <typename LhsPair, typename RhsPair>
  __device__ __forceinline__ 
  auto operator()(LhsPair const& lhs, RhsPair const& rhs) const noexcept
  {
    int right_join_key = get_from_batches(lhs.second, _dev_right_join_key_batch_iter,
                                          _right_batch_size, _num_right_batches);
    // return lhs.first == rhs.first;
    return lhs.first == rhs.first &&
           _left_join_key_iter[rhs.second] == right_join_key;
  }

private:
  int* _left_join_key_iter;
  int** _dev_right_join_key_batch_iter;
  row_size_t* _right_batch_size;
  int _num_right_batches;
};

class batch2_pair_equality
{
public:
  batch2_pair_equality(int** left_join_key_batch_iter,
                      row_size_t* left_batch_size,
                      int num_left_batches,
                      int** right_join_key_batch_iter,
                      row_size_t* right_batch_size,
                      int num_right_batches
                      )
    : _left_join_key_batch_iter(left_join_key_batch_iter),
      _left_batch_size(left_batch_size),
      _num_left_batches(num_left_batches),
      _right_join_key_batch_iter(right_join_key_batch_iter),
      _right_batch_size(right_batch_size),
      _num_right_batches(num_right_batches)
  {}

  template <typename LhsPair, typename RhsPair>
  __device__ __forceinline__ 
  auto operator()(LhsPair const& lhs, RhsPair const& rhs) const noexcept
  {
    int left_join_key = get_from_batches(rhs.second, _left_join_key_batch_iter,
                                        _left_batch_size, _num_left_batches);
    int right_join_key = get_from_batches(lhs.second, _right_join_key_batch_iter,
                                          _right_batch_size, _num_right_batches);
    return lhs.first == rhs.first && left_join_key == right_join_key;
  }

private:
  int** _left_join_key_batch_iter;
  row_size_t* _left_batch_size;
  int _num_left_batches;
  int** _right_join_key_batch_iter;
  row_size_t* _right_batch_size;
  int _num_right_batches;
};

class batch_groupby_key_hasher
{
public:
  batch_groupby_key_hasher(int** key0_batches,
                            int** key1_batches, 
                            row_size_t* key0_batches_size, 
                            row_size_t* key1_batches_size, 
                            int key0_num_batch,
                            int key1_num_batch)
    : dev_key0_batches_(key0_batches), 
      dev_key1_batches_(key1_batches), 
      key0_batches_size_(key0_batches_size),
      key1_batches_size_(key1_batches_size),
      key0_num_batch_(key0_num_batch),
      key1_num_batch_(key1_num_batch),
      num(2)
  {}
  batch_groupby_key_hasher(int** key0_batches,
                            int** key1_batches, 
                            int** key2_batches,
                            row_size_t* key0_batches_size, 
                            row_size_t* key1_batches_size, 
                            row_size_t* key2_batches_size,
                            int key0_num_batch,
                            int key1_num_batch,
                            int key2_num_batch)
    : dev_key0_batches_(key0_batches), 
      dev_key1_batches_(key1_batches), 
      dev_key2_batches_(key2_batches),
      key0_batches_size_(key0_batches_size),
      key1_batches_size_(key1_batches_size),
      key2_batches_size_(key2_batches_size),
      key0_num_batch_(key0_num_batch),
      key1_num_batch_(key1_num_batch),
      key2_num_batch_(key2_num_batch),
      num(3)
  {}

  __device__ int operator()(groupby_ht_keyT const key) const noexcept {
    int key0 = 0, key1 = 0, key2 = 0;
    key0 = get_from_batches(key.idx[0], dev_key0_batches_,
                            key0_batches_size_, key0_num_batch_);
    if (num >= 2) {
      key1 = get_from_batches(key.idx[1], dev_key1_batches_,
                              key1_batches_size_, key1_num_batch_);
    }
    if (num >= 3) {
      key2 = get_from_batches(key.idx[2], dev_key2_batches_,
                              key2_batches_size_, key2_num_batch_);
    }
    return key0 + key1 + key2;
    // return key0-1992 + key1*100;
  }
  
private:
  int num;
  int** dev_key0_batches_ = nullptr;
  int** dev_key1_batches_ = nullptr;
  int** dev_key2_batches_ = nullptr;
  row_size_t* key0_batches_size_ = nullptr;
  row_size_t* key1_batches_size_ = nullptr;
  row_size_t* key2_batches_size_ = nullptr;
  int key0_num_batch_ = 0;
  int key1_num_batch_ = 0;
  int key2_num_batch_ = 0;
};

class batch_groupby_key_equality
{
public:
  batch_groupby_key_equality(int** key0_batches,
                             int** key1_batches, 
                             row_size_t* key0_batches_size, 
                             row_size_t* key1_batches_size, 
                             int key0_num_batch,
                             int key1_num_batch)
    : dev_key0_batches_(key0_batches), 
      dev_key1_batches_(key1_batches), 
      key0_batches_size_(key0_batches_size),
      key1_batches_size_(key1_batches_size),
      key0_num_batch_(key0_num_batch),
      key1_num_batch_(key1_num_batch),
      num(2)
  {}
  batch_groupby_key_equality(int** key0_batches,
                             int** key1_batches, 
                             int** key2_batches,
                             row_size_t* key0_batches_size, 
                             row_size_t* key1_batches_size, 
                             row_size_t* key2_batches_size,
                             int key0_num_batch,
                             int key1_num_batch,
                             int key2_num_batch)
    : dev_key0_batches_(key0_batches), 
      dev_key1_batches_(key1_batches), 
      dev_key2_batches_(key2_batches),
      key0_batches_size_(key0_batches_size),
      key1_batches_size_(key1_batches_size),
      key2_batches_size_(key2_batches_size),
      key0_num_batch_(key0_num_batch),
      key1_num_batch_(key1_num_batch),
      key2_num_batch_(key2_num_batch),
      num(3)
  {}

  __device__ bool operator()(groupby_ht_keyT const lhs_key,
                             groupby_ht_keyT const rhs_key) const noexcept {
    // return lhs_key.d_idx == rhs_key.d_idx && lhs_key.p_idx == rhs_key.p_idx;
    // int key1_l = get_from_batches(lhs_key.idx0, dev_key1_batches_,
    //                               key1_batches_size_, key1_num_batch_);
    // int key1_r = get_from_batches(rhs_key.idx0, dev_key1_batches_,
    //                               key1_batches_size_, key1_num_batch_);
    // int key2_l = get_from_batches(lhs_key.idx1, dev_key2_batches_,
    //                               key2_batches_size_, key2_num_batch_);
    // int key2_r = get_from_batches(rhs_key.idx1, dev_key2_batches_,
    //                               key2_batches_size_, key2_num_batch_);
    // for (int i = 0; i < 4; i++) {
    //   if (lhs_key.idx[i] == row_idx_sentinel) {
    //     break;
    //   }
    //   const int keyi_l = get_from_batches(lhs_key.idx[i])
    // }
    const int key0_l = get_from_batches(lhs_key.idx[0], dev_key0_batches_,
                                        key0_batches_size_, key0_num_batch_);
    const int key0_r = get_from_batches(rhs_key.idx[0], dev_key0_batches_,
                                        key0_batches_size_, key0_num_batch_);
    if (key0_l != key0_r) {
      return false;
    }
    if (num >= 2) {
      const int key1_l = get_from_batches(lhs_key.idx[1], dev_key1_batches_,
                                          key1_batches_size_, key1_num_batch_);
      const int key1_r = get_from_batches(rhs_key.idx[1], dev_key1_batches_,
                                          key1_batches_size_, key1_num_batch_);
      if (key1_l != key1_r) {
        return false;
      }
    }
    if (num >= 3) {
      const int key2_l = get_from_batches(lhs_key.idx[2], dev_key2_batches_,
                                          key2_batches_size_, key2_num_batch_);
      const int key2_r = get_from_batches(rhs_key.idx[2], dev_key2_batches_,
                                          key2_batches_size_, key2_num_batch_);
      if (key2_l != key2_r) {
        return false;
      }
    }
    return true;
  }

private:
  int num;
  int** dev_key0_batches_ = nullptr;
  int** dev_key1_batches_ = nullptr;
  int** dev_key2_batches_ = nullptr;
  row_size_t* key0_batches_size_ = nullptr;
  row_size_t* key1_batches_size_ = nullptr;
  row_size_t* key2_batches_size_ = nullptr;
  int key0_num_batch_ = 0;
  int key1_num_batch_ = 0;
  int key2_num_batch_ = 0;
};

class make_pair_simple 
{
public:
  __host__ __device__ make_pair_simple(int* key)
    : _key(key)
  {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    // hash_value_type row_hash_value = (_key[i] - _min_key) % _ht_len;
    hash_value_type row_hash_value = _key[i];
    return cuco::make_pair(row_hash_value, size_type{i});
  }

private:
  int* _key;
};

// struct cond_base
// {
//   cond_base() {}
//   virtual __host__ __device__
//       bool operator()(const int& in) const = 0;
// };
// struct p_category_cond : cond_base
// {
//     p_category_cond() {}
//     __host__ __device__
//       bool operator()(const int& c) const override {
//           return c == 1;      // 4%
//           // return c >= 0 && c <= 3;    // 16%
//           // return c%4 == 0;      // 28%
//           // return c%3 == 0;      // 36%
//           // return c%2 == 0;      // 52%
//           // return c >= 8;    // 68%
//           // return c >= 5;    // 80%
//       }
// };
// struct s_region_cond : cond_base
// {
//     s_region_cond() {}
//     __host__ __device__
//       bool operator()(const int& r) const override {
//           return r==1;      // 20%
//           // return r==0 || r==1;      // 40%
//           // return r!=1;      // 80%
//           // return true;      // 100%
//       }
// };
// struct d_cond : cond_base
// {
//   d_cond() {}
//   __host__ __device__
//     bool operator()(const int& r) const override {
//       return true;
//     }
// };