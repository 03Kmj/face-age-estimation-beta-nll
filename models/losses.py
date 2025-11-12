import torch
import torch.nn as nn

from torch.distributions.beta import Beta

class Beta_NLL_Loss(nn.Module):
    
    def __init__(self, beta_reweight=0.5, epsilon=1e-6):
        super(Beta_NLL_Loss, self).__init__()
        self.beta_reweight = beta_reweight 
        self.epsilon = epsilon 

    def forward(self, pred_alpha, pred_beta, target_y):
        
        alpha = pred_alpha + self.epsilon
        beta = pred_beta + self.epsilon
        
        # 정답 레이블을 (0, 1) 사이로 클리핑하여 수치 안정성 확보
        target_y_stable = torch.clamp(target_y, self.epsilon, 1.0 - self.epsilon)
        
        # Beta 분포 객체 생성 및 Log-Likelihood 계산
        m = Beta(alpha, beta)
        log_likelihood = m.log_prob(target_y_stable)
        NLL = -log_likelihood
        
        # 분산(V)을 이용한 재가중치 (Reweighting)
        V = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        V_stable = torch.clamp(V, self.epsilon, 1e5) 
        
        reweighting_factor = torch.pow(V_stable, -self.beta_reweight)
        
        # 최종 Beta-NLL 손실: NLL * 재가중치
        beta_nll_loss = (NLL * reweighting_factor).mean()

        return beta_nll_loss