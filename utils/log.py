import os
import wandb


class Wandb:
    def __init__(self, args, api_key):
        """
        Khởi tạo Wandb logger.

        Args:
            args: Đối tượng argparse.Namespace chứa các tham số cấu hình.
            wandb_api_key: Wandb API key từ biến môi trường.
        """
        self.args = args
        self.api_key = api_key
        self.project = args.model
        self.session = args.session_id
        # Đăng nhập Wandb
        wandb.login()

        # Khởi tạo Wandb run
        wandb.init(
            project= self.project,
            name= self.session,
            config=vars(args),
            reinit=True
        )

    def log_metrics(self, metrics):
        """Ghi metrics vào Wandb."""
        wandb.log(metrics)

    def save_model(self, path):
        """Lưu file checkpoint vào Wandb."""
        wandb.save(path)

    def finish(self):
        """Kết thúc Wandb run."""
        wandb.finish()

