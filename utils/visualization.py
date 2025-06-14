from tabulate import tabulate
from colorama import init, Fore, Back, Style
import argparse

def print_eval( train_loss, val_loss, test_loss, round_digits):
    print(f"Train Loss: {train_loss:.{round_digits}f} Val Loss: {val_loss:.{round_digits}f} Test Loss: {test_loss:.{round_digits}f}")

def display_args_table(args, accelerator=None):
    """
    Hiển thị các tham số từ args dưới dạng bảng trong terminal với màu chữ xen kẽ và highlight header.
    
    Args:
        args: Đối tượng argparse.Namespace chứa các tham số.
        accelerator: Đối tượng Accelerate (nếu có) để kiểm tra tiến trình chính.
    """
    if accelerator is None or accelerator.is_local_main_process:
        # Chuyển args thành danh sách các cặp (key, value)
        args_dict = vars(args)
        table_data = [[key, value] for key, value in args_dict.items()]
        
        # Định dạng header với màu vàng, chữ đậm
        headers = [
            f"{Fore.YELLOW}{Style.BRIGHT}Parameter{Style.RESET_ALL}",
            f"{Fore.YELLOW}{Style.BRIGHT}Value{Style.RESET_ALL}"
        ]
        
        # Tạo bảng với màu chữ xen kẽ
        colored_table_data = []
        for i, row in enumerate(table_data):
            # Màu chữ xen kẽ: CYAN cho dòng chẵn, WHITE cho dòng lẻ
            text_color = Fore.CYAN if i % 2 == 0 else Fore.WHITE
            colored_row = [
                f"{text_color}{row[0]}{Style.RESET_ALL}",
                f"{text_color}{row[1]}{Style.RESET_ALL}"
            ]
            colored_table_data.append(colored_row)
        
        # Hiển thị tiêu đề và bảng
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Arguments Table:{Style.RESET_ALL}")
        print(tabulate(colored_table_data, headers=headers, tablefmt="grid"))