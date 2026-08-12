"""CLI argument parsing for the geocoreg package."""

import argparse

from geocoreg import pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="geocoreg",
        description="Dense Displacement Co-Registration Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments for both subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ref", required=True, help="Path to reference raster")
    common.add_argument("--mov", required=True, help="Path to moving raster")
    common.add_argument("--out", required=True,
                        help="Output folder (result written as matched.tif)")

    # match subcommand
    subparsers.add_parser("match", parents=[common],
                          help="Full pipeline: match + warp")

    # apply subcommand
    apply_parser = subparsers.add_parser("apply", parents=[common],
                                         help="Apply existing control points to warp")
    apply_parser.add_argument("--cp", required=True,
                              help="Path to existing control point file (.geojson)")

    args = parser.parse_args()

    if args.command == "match":
        pipeline.run_match(args.ref, args.mov, args.out)
    elif args.command == "apply":
        pipeline.run_apply(args.ref, args.mov, args.out, args.cp)
