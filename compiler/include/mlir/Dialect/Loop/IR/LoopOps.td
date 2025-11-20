#ifndef MLIR_DIALECT_LOOP_IR_OPS
#define MLIR_DIALECT_LOOP_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Loop/IR/LoopInterfaces.td"

// include "mlir/Dialect/RelAlg/IR/RelAlgOps.td"

def Loop_Dialect : Dialect {
	let name = "loop";
    let summary = "A dialect for looped relational algebra operators";
    let cppNamespace = "::mlir::loop";
    let extraClassDeclaration = [{
        private:
        void registerTypes();

        public:

    }];
}


class Loop_Op<string mnemonic, list<Trait> traits = []> : Op<Loop_Dialect, mnemonic, traits> {
    // let hasCustomAssemblyFormat = 1;
}
class Loop_Type<string name, string typeMnemonic> : TypeDef<Loop_Dialect, name> {
    let mnemonic = typeMnemonic;
}

// class Loop_Collection<string name, string typeMnemonic> 
//         : TypeDef<Loop_Dialect, name, [], "::mlir::loop::CollectionType"> {
//     let mnemonic = typeMnemonic;
// }

// def Loop_Collection : Type<CPred<"$_self.isa<::mlir::loop::CollectionType>()">, "DSA dialect collection">;

def Loop_JoinHashtable : Loop_Type<"JoinHashtable","join_ht"> {
    let summary = "join hashtable";
}

def Loop_AggregationHashtable : Loop_Type<"AggregationHashtable","aggr_ht"> {
    let summary = "aggregation hashtable";
}

def Loop_ProbeResult : Loop_Type<"ProbeResult", "probe_res"> {
}



def Loop_CreateHashTable : Loop_Op<"create_ht"> {
    let summary = "Create the hash table for hash join";
    let results = (outs AnyType:$ht);
    let arguments = (ins ArrayAttr:$keys, ArrayAttr:$payloads);
}

def Loop_UpdateHashTable : Loop_Op<"update_ht", [Terminator]> {
    let summary = "Build the hash table for hash join";
    let arguments = (ins AnyType:$ht, AnyType:$tuple, ArrayAttr:$keys, ArrayAttr:$payloads);
}

def Loop_ProbeHashTable : Loop_Op<"probe_ht"> {
    let summary = "Probe the hash table for hash join";
    let arguments = (ins AnyType:$ht, AnyType:$tuple, ArrayAttr:$keys, ArrayAttr:$payloads);
    let results = (outs Loop_ProbeResult:$res);
}

def ForOp : Loop_Op<"for", [SingleBlock]> {
    // Type: [0: join build/selection, 1: probe, 2: aggregation build, 3: retrieve, 4: map]
    let arguments = (ins AnyType:$table, ArrayAttr:$req_cols, StrAttr:$name_inp, StrAttr:$name_res, I8Attr:$type);
    let regions = (region SizedRegion<1>:$region);
    let results = (outs Optional<AnyType>:$res);
}

def IfOp : Loop_Op<"if", [SingleBlock, Terminator]> {
    let arguments = (ins AnyType:$cond);
    let regions = (region SizedRegion<1>:$region);
}

def UpdateOp : Loop_Op<"update", [ParentOneOf<["ForOp", "IfOp"]>]> {
    let arguments = (ins AnyType:$input, Variadic<AnyType>:$values, ArrayAttr:$cols);
    let results = (outs AnyType:$output);
}

def YieldOp : Loop_Op<"yield", [NoSideEffect, Terminator,
                                ParentOneOf<["ForOp", "IfOp"]>]> {
    let arguments = (ins AnyType:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Loop_Shuffle : Loop_Op<"shfl"> {
    let arguments = (ins AnyType:$tbl);
}

#endif
