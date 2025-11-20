#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "arrow/array.h"
#include "mlir-support/eval.h"
#include "runner/runner.h"
#include "runtime/MetaDataOnlyDatabase.h"

void check(bool b, std::string message) {
   if (!b) {
      std::cerr << "ERROR: " << message << std::endl;
      exit(1);
   }
}

int main(int argc, char** argv) {
   std::string inputFileName = std::string(argv[1]);
   std::ifstream istream{inputFileName};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   std::string sqlQuery = buffer.str();
   runtime::ExecutionContext context;
   context.id = 42;
   if (argc <= 2) {
      std::cerr << "USAGE: dhap-compiler *.sql <database_dir>" << std::endl;
      return 1;
   }
   auto db_path = std::string(argv[2]);
   // auto database = runtime::Database::loadFromDir(db_path);
   auto database = runtime::Database::loadFromDirSample(db_path);
   context.db = std::move(database);

   support::eval::init();
   runner::RunMode runMode = runner::Runner::getRunMode();
   runner::Runner runner(runMode);
   check(runner.loadSQL(sqlQuery, *context.db), "SQL translation failed");
   check(runner.dhap_compile(*context.db), "query optimization failed");

   return 0;
}
