# Out of Docker
```
git remote add onr ssh://zcq@r6:/home/zcq/versa-cube
```

## gdrcopy (v2.4.1)
```
pushd gdrcopy && mkdir install
make prefix=/home/zcq/gdrcopy/install CUDA=/usr/local/cuda-11.8 all install
popd && docker cp gdrcopy vc-$HOSTNAME:/workspace/
```


# In Docker

## SSH
```
apt install -y openssh-server
sed -i 's/#Port 22/Port 2233/' /etc/ssh/sshd_config
/etc/init.d/ssh start
useradd -m -p $(echo test | openssl passwd -1 -stdin) -s /bin/bash test
su test
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
# update .ssh/authorized_keys
# update .ssh/config
printf "Host r1d\n  HostName r1\n  Port 2233\nHost r4d\n  HostName r4\n  Port 2233\nHost r5d\n  HostName r5\n  Port 2233\nHost r6d\n  HostName r6\n  Port 2233\n" > .ssh/config
```

## Network Card Driver
```
apt install ./linux-headers-5.15.102_5.15.102-1_amd64.deb		# for modified kernel...
# apt install libnvidia-compute-545
mount -o ro,loop MLNX_OFED_LINUX-23.10-2.1.3.1-ubuntu22.04-x86_64.iso /mnt
cd /mnt && ./mlnxofedinstall -q
/etc/init.d/openibd restart
```

## UCX (v1.16.0)
```
./autogen.sh && ./contrib/configure-release  --with-cuda=/usr/local/cuda-11.8 --with-gdrcopy=/workspace/gdrcopy/install --enable-mt
make -j32 && make install
```

## OpenMPI (v4.1.5)
```
./configure --with-cuda=/usr/local/cuda-11.8 --with-ucx=/usr
make -j32 && make install
```

## MPI Benchmark
### Install
```
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
tar -xvf osu-micro-benchmarks-7.4.tar.gz && cd osu-micro-benchmarks-7.4
./configure CC=mpicc CXX=mpicxx  --enable-cuda --with-cuda=/usr/local/cuda-11.8
make -j32 && make install
```

### Running
```
sed -i '2i\
* hard memlock unlimited\
* soft memlock unlimited' /etc/security/limits.conf

mpirun -host r6d,r4d -np 2 -npernode 1 -mca pml ucx -bind-to core  -x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_RNDV_SCHEME=get_zcopy /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw  D D

mpirun -mca pml ucx -host r6d -n 1 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_RNDV_SCHEME=put_ppln /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw  H D : -host r4d -n 1 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_RNDV_SCHEME=put_ppln /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw  H D

mpirun -host r6d -mca pml ucx -bind-to core -x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_RNDV_SCHEME=get_zcopy /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw  D D : -host r1d -mca pml ucx -bind-to core -x CUDA_VISIBLE_DEVICES=0 -x UCX_NET_DEVICES=mlx5_1:1 -x UCX_RNDV_SCHEME=get_zcopy /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw  D D
```

## For convenience
```
chown -R test:test build/
export PATH=/workspace/versa-cube/build/:$PATH
ssh r4d "echo "export PATH=/workspace/versa-cube/build/:$PATH" >> /home/test/.bashrc "
```
```
DHAP_PLAN=/g2./g2./g2./g2 bash run.sh --server_ip r6 --plan_dir /home/test/test1 --data_dir /workspace/data/arrow/ssb_1000i/ --sql_file /workspace/versa-cube/resources/sql/ssb/42.sql --dist --max_numb 16 --max_shflw 4

mem_server --data_dir /workspace/data/arrow/ssb_1000i/ --load ssb
```


## For large data
```
split --additional-suffix=.tbl -a 1 -d -n l/3 lineorder.tbl lineorder-

```