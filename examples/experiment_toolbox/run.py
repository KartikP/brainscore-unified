"""Compatibility entry point for the independently installed example package.

Install examples/partner_tool first, then run this script or use:
python -I -m umi_partner_tool.toolbox --out NEW_DIRECTORY
"""

from umi_partner_tool.toolbox import main


if __name__ == '__main__':
    main()
