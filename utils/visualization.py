from tabulate import tabulate
import argparse

def display_args_table(args):
    """
    Hiển thị các tham số từ args dưới dạng bảng trong terminal.
    
    Args:
        args: Đối tượng argparse.Namespace chứa các tham số.
    """
    # Chuyển args thành danh sách các cặp (key, value)
    args_dict = vars(args)  # Chuyển Namespace thành dict
    table_data = [[key, value] for key, value in args_dict.items()]
    
    # Tạo bảng với tabulate
    headers = ["Parameter", "Value"]
    print("\nArguments Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))