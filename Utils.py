import matplotlib.pyplot as plt
import numpy as np
import torch


def saveModel(model:torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def loadModel(model:torch.nn.Module, path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    return model


def copyModel(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


def exportONNX(model: torch.nn.Module, sample_inputs: list[torch.Tensor], path: str) -> None:
    model.eval()
    torch.onnx.export(model, sample_inputs, path, verbose=True, opset_version=13, 
                      input_names=['input_traj', "time", "attr"], output_names=['output_traj'])
    

def visualizeTraj(lon_lat: torch.Tensor, times: torch.Tensor, draw_dot: bool = True) -> None:
    """ draw trajectory

    :param lon_lat: lon_lat trajectory (2, traj_length)
    :param times: time of each point (traj_length)
    :return: None
    """
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.plot(lon_lat[0, :].cpu(), lon_lat[1, :].cpu(), color='#101010', linewidth=0.1)
    if draw_dot:
        plt.scatter(lon_lat[0, :].cpu(), lon_lat[1, :].cpu(), c=times.cpu(), cmap='rainbow', s=0.5)


def renderTrajRecovery(traj: torch.Tensor, recover_traj: torch.Tensor, no_noise_recovery: torch.Tensor = None) -> plt.figure:
    # traj: (2, traj_length)
    # recover_traj: (2, traj_length)

    # draw original trajectory
    plt.subplot(1, 3, 1)
    plt.title("original")
    visualizeTraj(traj.detach(), torch.arange(traj.shape[1]))

    # draw recovered trajectory
    plt.subplot(1, 3, 2)
    plt.title("rec(noise)")
    visualizeTraj(recover_traj.detach(), torch.arange(traj.shape[1]))

    plt.subplot(1, 3, 3)
    plt.title("rec(no noise)")
    visualizeTraj(no_noise_recovery.detach(), torch.arange(traj.shape[1]))

    # render the figure and return the image as numpy array
    plt.tight_layout()

    return plt.gcf()



class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window = np.zeros(window_size, dtype=np.float32)
        self.window_size = window_size
        self.cursor = 0

    def __lshift__(self, number: float) -> None:
        self.window[self.cursor] = number
        self.cursor = (self.cursor + 1) % self.window_size

    def __str__(self) -> str:
        return str(self.window.mean())
    
    def __repr__(self) -> str:
        return str(self.window.mean())
    
    def __format__(self, format_spec: str) -> str:
        return self.window.mean().__format__(format_spec)

    def __float__(self) -> float:
        return float(self.window.mean())


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, model_cfg: dict) -> None:
        self.decay = decay
        self.remain = 1 - decay
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
        self.ema_model = model.__class__(**model_cfg)

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                mov_avg = self.decay * self.shadow_params[name] + self.remain * param.data
                self.shadow_params[name] = mov_avg.clone()


    def getModel(self):
        for name, param in self.ema_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow_params[name].data)
        return self.ema_model


# if __name__ =="__main__":
#     import cv2
#     traj = torch.arange(0, 10, 0.1).reshape(2, 50) + torch.randn(2, 50) * 0.1
#     recover_traj = traj + torch.randn(2, 50) * 0.3
#     plot = renderTrajRecovery(traj, recover_traj)
#     plt.show()
#
#     cv2.imshow("test", plot)
#     cv2.waitKey(0)