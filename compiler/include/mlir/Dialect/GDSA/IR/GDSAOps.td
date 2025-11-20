#ifndef MLIR_DIALECT_GDSA_IR_OPS
#define MLIR_DIALECT_GDSA_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Dialect/GDSA/IR/GDSAInterfaces.td"

// include "mlir/Dialect/RelAlg/IR/RelAlgOps.td"

def GDSA_Dialect : Dialect {
	let name = "GDSA";
    let summary = "A dialect for GPU Data Structure and Algorithm";
    let cppNamespace = "::mlir::gdsa";
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:

    }];
}


class GDSA_Op<string mnemonic, list<Trait> traits = []> : Op<GDSA_Dialect, mnemonic, traits> {
	// let hasCustomAssemblyFormat = 1;
}
class GDSA_Type<string name, string typeMnemonic> : TypeDef<GDSA_Dialect, name> {
	let mnemonic = typeMnemonic;
}


def GDSA_RowIndex : GDSA_Type<"RowIndex", "row_idx"> {
}

def GDSA_Pair : GDSA_Type<"Pair", "pair"> {
}

def GDSA_HashValue : GDSA_Type<"HashValue", "hash_value"> {
}

def GDSA_Map : GDSA_Type<"Map","map"> {
}

def GDSA_MultiMap : GDSA_Type<"MultiMap","multi_map"> {
}

def GDSA_AggrKey : GDSA_Type<"AggrKey","aggr_key"> {
}


def GDSA_MakeAggrKey : GDSA_Op<"make_aggr_key"> {
	let summary = "Make an aggregated key (for hash aggregation)";
	let arguments = (ins Variadic<AnyType>:$inputs);
	let results = (outs GDSA_AggrKey:$res);
}

def GDSA_GetElement : GDSA_Op<"get"> {
	let summary = "Get an element from an column";
	let arguments = (ins AnyAttr:$col, GDSA_RowIndex:$idx);
	let results = (outs AnyType:$elem);
}

def GDSA_StoreElement : GDSA_Op<"store"> {
	let summary = "Get an element from an column";
	let arguments = (ins AnyType:$value, AnyAttr:$col, GDSA_RowIndex:$idx);
}

def GDSA_MakePair : GDSA_Op<"make_pair"> {
	let summary = "Get an pair of key and value";
	let arguments = (ins AnyType:$key, AnyType:$val);
	let results = (outs GDSA_Pair:$pair);
}

def GDSA_GetSecond : GDSA_Op<"get_second"> {
	let summary = "Get the value of (key, value) pair";
	let arguments = (ins GDSA_Pair:$pair);
	let results = (outs AnyType:$sec);
}

def GDSA_CreateMap : GDSA_Op<"create_map"> {
	let summary = "Create a (multi) map";
	let results = (outs AnyType:$map);
}

def GDSA_InsertMap : GDSA_Op<"insert_map"> {
	let summary = "Insert to (multi) map";
	let arguments = (ins AnyType:$map, AnyType:$pair, ArrayAttr:$payloads);
}

def GDSA_MapCount: GDSA_Op<"map_count"> {
	let summary = "Count in (multi) map";
	let arguments = (ins AnyType:$map, AnyType:$pair, ArrayAttr:$keys, ArrayAttr:$payloads);
	let results = (outs I8:$count);
}

def GDSA_MapFind: GDSA_Op<"map_find", [NoSideEffect]> {
	let summary = "Find in a (multi) map";
	let arguments = (ins AnyType:$map, AnyType:$pair);
	let results = (outs AnyType:$res);
}

def GDSA_While : GDSA_Op<"while", [SingleBlock]> {
	let summary = "While";
	let arguments = (ins StrAttr:$size, ArrayAttr:$req_cols);
	let regions = (region SizedRegion<1>:$region);
}

def GDSA_Increase : GDSA_Op<"increase", [Terminator]> {
	let arguments = (ins GDSA_RowIndex:$idx);
	// let arguments = (ins GDSA_RowIndex:$idx, I32Attr:$stride);
}

def GDSA_Equal : GDSA_Op<"eq"> {
	let arguments = (ins AnyType:$left, AnyAttr:$right);
	let results = (outs I1:$res);
}

def GDSA_NotEqual : GDSA_Op<"neq"> {
	let arguments = (ins AnyType:$left, Optional<AnyType>:$right, AnyAttr:$right1);
	let results = (outs I1:$res);
}

def GDSA_If : GDSA_Op<"if", [SingleBlock]> {
	let arguments = (ins AnyTypeOf<[I1, I32]>:$cond);
	let regions = (region SizedRegion<1>:$region);
}

def GDSA_End : GDSA_Op<"end", [Terminator]> {
	let summary = "To terminate if";
}

def GDSA_Const : GDSA_Op<"const"> {
	let arguments = (ins AnyAttr:$v);
	let results = (outs AnyType:$res);
}

class GDSA_BinaryOp<string mnemonic, list<Trait> traits = []> :
	Op<GDSA_Dialect, mnemonic,
			!listconcat(traits, [NoSideEffect])>, Arguments<(ins I32:$left, I32:$right)> {
  let results = (outs I32:$result);
}

def GDSA_Add : GDSA_BinaryOp<"add"> {}
def GDSA_Sub : GDSA_BinaryOp<"sub"> {}
def GDSA_Mul : GDSA_BinaryOp<"mul"> {}

def GDSA_AtomicAdd : GDSA_Op<"atomic_add"> {
	let arguments = (ins AnyAttr:$col, AnyType:$idx, AnyType:$val);
	let results = (outs Optional<AnyType>:$res);
}

def GDSA_AtomicAdd1 : GDSA_Op<"atomic_add1"> {
	let arguments = (ins AnyAttr:$col, AnyType:$idx);
	let results = (outs Optional<AnyType>:$res);
}

def GDSA_Mod : GDSA_Op<"mod"> {
	let arguments = (ins AnyType:$left, AnyAttr:$right);
	let results = (outs AnyType:$res);
}

def GDSA_GetGlobalIndex : GDSA_Op<"get_global_idx"> {
	let arguments = (ins I32:$p_id, GDSA_RowIndex:$idx_in_p);
	let results = (outs GDSA_RowIndex:$global_idx);
}

def GDSA_Materialize : GDSA_Op<"materialize"> {
	let arguments = (ins GDSA_RowIndex:$pos, Variadic<AnyType>:$results);
}

def GDSA_SingleThread : GDSA_Op<"single", [SingleBlock]> {
	let regions = (region SizedRegion<1>:$region);
}

def GDSA_AnyThread : GDSA_Op<"any"> {
	let arguments = (ins I1:$input);
	let results = (outs I32:$res);
}

def GDSA_CreateResult : GDSA_Op<"create_res"> {
	let results = (outs AnyType:$res);
}

#endif
