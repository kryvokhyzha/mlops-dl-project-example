from config import Config
from utils import generate_sample


if __name__ == "__main__":
    opt = Config()
    df = generate_sample("multiclass_classification", n_classes=3)
    df.to_csv(opt.path_to_data / "sample.csv", index=False, sep=";")
