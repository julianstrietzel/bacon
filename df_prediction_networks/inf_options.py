from argparse import Namespace

from options.base_options import BaseOptions


class InferenceOptions(BaseOptions):
    def __init__(self, opt_path):
        super(InferenceOptions, self).__init__()
        self.opt_path = opt_path
        self.opt = Namespace()
        self.initialized = False
        self.is_train = False

    def initialize(self):
        super().initialize()
        # self.opt, _ = self.parser.parse_known_args()
        for line in open(self.opt_path, "r"):
            x = line.strip().split(":")
            if len(x) != 2:
                continue
            opt_id = x[0].strip()
            if opt_id == "expr_dir":
                reworked_expr_dir = (
                    f"../df_prediction_networks/{BaseOptions.EXTRACTOR_CLASS}"
                    + x[1].strip()[1:]
                )
                setattr(self.opt, opt_id, reworked_expr_dir)
                continue
            setattr(self.opt, opt_id, convert(x[1].strip()))
        # set is inference flag
        setattr(self.opt, "is_inference", True)
        self.opt.is_train = self.is_train

    def parse(self):
        if not self.initialized:
            self.initialize()
        return self.opt


def convert(txt):
    # check if txt is list
    str_to_val = {"": None, "True": True, "False": False}
    if txt in str_to_val:
        return str_to_val[txt]
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
        txt = txt.split(",")
        txt = [convert(x.strip()) for x in txt]
        return txt
    try:
        k = float(txt)
        if k.is_integer():
            return int(k)
        return k
    except ValueError:
        return str(txt)
