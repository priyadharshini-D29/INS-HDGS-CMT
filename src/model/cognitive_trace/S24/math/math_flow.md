
# INS-HDGS-CMT Mathematical Flow

## EEG Input
X \in R^{1500 x 24}

## Normalization
X' = (X - mu)/sigma

## SNN
tau dV/dt = -V + I

Spike:
S(t)=H(V(t)-Vth)

## Graph Attention
alpha_ij =
exp(LeakyReLU(aT[Wh_i||Wh_j]))
/
sum_k exp(...)

## Transformer
Attention(Q,K,V)=softmax(QKT/sqrt(dk))V

## Contrastive Loss
L=-log(exp(sim(zi,zj)/tau)/sum(exp(...)))

## MMD
MMD(Xs,Xt)=||mean(phi(Xs))-mean(phi(Xt))||^2

## Fusion
Zfusion = Concat(Zeeg,Zet)

## Softmax
P(y)=exp(z_i)/sum(exp(z_j))
