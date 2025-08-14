import torch
import copy

class EarlyStopper:
    """
    훈련 중 검증 손실이 개선되지 않을 때 훈련을 조기 종료하는 클래스.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        :param patience: 성능 개선이 없을 때 몇 번의 에포크를 더 기다릴지.
        :param min_delta: 개선으로 간주할 최소한의 변화량.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        매 에포크마다 호출되어 조기 종료 여부를 판단합니다.
        
        :param val_loss: 현재 에포크의 검증 손실.
        :param model: 현재 훈련 중인 PyTorch 모델.
        :return: 조기 종료가 필요한 경우 True, 아니면 False.
        """
        if val_loss < self.best_loss - self.min_delta:
            # 손실이 개선된 경우
            self.best_loss = val_loss
            self.counter = 0
            # 최적 모델 상태 저장
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        else:
            # 손실이 개선되지 않은 경우
            self.counter += 1
            print(f"조기 종료 카운터: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"조기 종료: {self.patience} 에포크 동안 성능 개선이 없어 훈련을 중단합니다.")
                return True
            return False

    def load_best_model(self, model):
        """
        가장 좋았던 성능을 보인 모델 상태를 로드합니다.
        
        :param model: 상태를 로드할 PyTorch 모델.
        """
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
            print("최적의 모델 상태로 복원 완료.")