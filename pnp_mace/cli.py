"""Console script for pnp_mace."""
import argparse
import sys


def main():
    """Console script for pnp_mace."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "pnp_mace.cli.main")
    return 0


# TODO:  implement the CLI
if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
