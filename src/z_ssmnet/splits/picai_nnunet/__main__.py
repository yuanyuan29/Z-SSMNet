from z_ssmnet.splits import export_splits
from z_ssmnet.splits.picai_nnunet import (nnunet_splits, train_splits, valid_splits)

if __name__ == "__main__":
    export_splits(
        train_splits=train_splits,
        valid_splits=valid_splits,
        nnunet_splits=nnunet_splits,
    )
