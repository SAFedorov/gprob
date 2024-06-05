# The script for the generation of binary operations for SparseNormal

def gen_op_method(name: str):
    op_dict = {"add": "+", 
                "sub": "-", 
                "mul": "*", 
                "truediv": "/", 
                "pow": "**"}
    
    if name.startswith("__r"):
        op = op_dict[name[3:-2]]
        vln = f"v = other.v {op} self.v"
    else:
        op = op_dict[name[2:-2]]
        vln = f"v = self.v {op} other.v"

    code = f"""
    def {name}(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented
        
        {vln}
        iaxes = _validate_iaxes([self, other])
        return SparseNormal(v, iaxes)
    """
    return code

method_names = ["__add__", "__radd__", 
                "__sub__", "__rsub__", 
                "__mul__", "__rmul__", 
                "__truediv__", "__rtruediv__", 
                "__pow__", "__rpow__"]

print("".join(gen_op_method(name) for name in method_names))
