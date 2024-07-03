# The script for the generation of binary operations for SparseNormal

def gen_op_method(name: str):
    op_dict = {"add": "+", 
                "sub": "-", 
                "mul": "*", 
                "truediv": "/", 
                "pow": "**"}
    
    if name.startswith("__r"):
        op = op_dict[name[3:-2]]
        vln = f"fcv = other.fcv {op} self.fcv"
        nlns = "other = assparsenormal(other)"
    else:
        op = op_dict[name[2:-2]]
        vln = f"fcv = self.fcv {op} other.fcv"
        nlns = """try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented
        """

    code = f"""
    def {name}(self, other):
        {nlns}
        {vln}
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)
    """
    return code


def gen_all():
    method_names = ["__add__", "__radd__", 
                    "__sub__", "__rsub__", 
                    "__mul__", "__rmul__", 
                    "__truediv__", "__rtruediv__", 
                    "__pow__", "__rpow__"]

    return "".join(gen_op_method(name) for name in method_names)


if __name__ == "__main__":
    print(gen_all())