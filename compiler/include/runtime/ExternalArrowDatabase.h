
#ifndef RUNTIME_EXTERNALARROWDATABASE_H
#define RUNTIME_EXTERNALARROWDATABASE_H
#include "runtime/Database.h"
#include <assert.h>

namespace runtime {
class ExternalArrowDatabase : public Database{
   std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tables;

   public:
   std::shared_ptr<arrow::Table> getTable(const std::string& name) override;
   std::shared_ptr<arrow::RecordBatch> getSample(const std::string& name) override;
   std::shared_ptr<TableMetaData> getTableMetaData(const std::string& name) override;
   void addTable(std::string, std::shared_ptr<arrow::Table>);
   bool hasTable(const std::string& name) override;
   void addATable(std::string table_name, std::shared_ptr<arrow::Table> table) override {
      assert(false && "should not be called");
   }
};
} // end namespace runtime
#endif // RUNTIME_EXTERNALARROWDATABASE_H


