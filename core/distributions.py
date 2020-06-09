import numpy as np
import torch
import torch.nn.functional as F

from torch import distributions


class MOGTwo:
    def __init__(self, device):
        self.num_dims = 2
        self.var = 0.5
        self.cov_matrix = torch.from_numpy(self.var * np.eye(2)).to(device).float()
        self.mean1 = torch.Tensor([0, 0]).to(device).float()
        self.mean2 = torch.Tensor([2, 2]).to(device).float()

    def log_prob(self, x):
        prob1 = torch.exp(-torch.sum((x - self.mean1) ** 2, dim=1))
        prob2 = torch.exp(-torch.sum((x - self.mean2) ** 2, dim=1))
        return torch.log(prob1 + prob2+1e-10)

    def mean(self):
        return (0.5*(self.mean1+self.mean2)).cpu().numpy()

    def std(self):
        mu = self.mean()
        variance = self.var + 0.5*(self.mean1.cpu().numpy()**2 + self.mean2.cpu().numpy()**2) - mu**2
        return np.sqrt(variance)

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-8, 8]

    @staticmethod
    def ylim():
        return [-8, 8]


class MOG:
    def __init__(self, means, cov_matrix, device):
        self.num_dims = 2
        self.g = []
        cov_matrix = torch.from_numpy(cov_matrix).to(device).float()
        for mean in means:
            mean = torch.Tensor(mean).to(device).float()
            self.g.append(distributions.multivariate_normal.MultivariateNormal(mean, cov_matrix))

    def log_prob(self, x):
        prob = 0.0
        for g in self.g:
            prob += torch.exp(g.log_prob(x)) / len(self.g)
        return torch.log(prob + 1e-10)


class Ring:
    def __init__(self, device):
        self.num_dims = 2
        self.device = device

    def energy(self, x):
        assert x.shape[1] == 2
        return (torch.sqrt(torch.sum(x ** 2, dim=1)) - 2.0) ** 2 / 0.32

    def log_prob(self, x):
        return -self.energy(x)

    def mean(self):
        return np.array([0., 0.])

    def std(self):
        return np.array([1.497, 1.497])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-4, 4]

    @staticmethod
    def ylim():
        return [-4, 4]


class Rings:
    def __init__(self, device):
        self.num_dims = 2
        self.device = device

    def energy(self, x):
        assert x.shape[1] == 2
        p1 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 1.0) ** 2 / 0.04
        p2 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 2.0) ** 2 / 0.04
        p3 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 3.0) ** 2 / 0.04
        p4 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 4.0) ** 2 / 0.04
        p5 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 5.0) ** 2 / 0.04
        return torch.min(torch.min(torch.min(torch.min(p1, p2), p3), p4), p5)

    def log_prob(self, x):
        return -self.energy(x)

    @staticmethod
    def mean():
        return np.array([3.6])

    @staticmethod
    def std():
        return np.array([1.24])

    @staticmethod
    def xlim():
        return [-6, 6]

    @staticmethod
    def ylim():
        return [-6, 6]

    @staticmethod
    def statistics(z):
        z_ = torch.sqrt(torch.sum(z**2, dim=1, keepdims=True))
        return z_


class BayesianLogisticRegression:
    def __init__(self, data, labels, device):
        self.data = torch.tensor(data).to(device).float()
        self.labels = torch.tensor(labels).to(device).float().flatten()
        self.num_features = self.data.shape[1]
        self.num_dims = self.num_features + 1

    def view_params(self, v):
        w = v[:,:self.num_features].view([-1, self.num_features, 1])
        b = v[:,self.num_features:].view([-1, 1, 1])
        return w, b

    def energy(self, v):
        w, b = self.view_params(v)
        x = self.data
        y = self.labels.view([1,-1,1])
        logits = torch.matmul(x,w) + b
        probs = torch.nn.Sigmoid()(logits)
        nll = -y*torch.log(probs + 1e-16) - (1.0-y)*torch.log(1.0-probs  + 1e-16)
        nll = torch.sum(nll, dim=[1,2])
        negative_logprior = torch.sum(0.5*w**2/0.1, dim=[1,2])
        return negative_logprior + nll

    def log_prob(self, v):
        return -self.energy(v)


class Australian(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/australian/data.npy')
        labels = np.load('../data/australian/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Australian, self).__init__(data, labels, device)

        
    def mean(self):
        return np.array([0.00251228,  0.05158003, -0.07079323,  0.29524441,  0.59452954,
                         0.11131982,  0.2354203 ,  1.43293823,  0.28597028,  0.49290017,
                         -0.10600186,  0.13054036, -0.25054757,  0.58509614, -0.38276327])
    
    def std(self):
        return np.array([0.11082   , 0.11738931, 0.11642496, 0.11251291, 0.12473013,
                         0.12270894, 0.13582683, 0.12235203, 0.13080647, 0.18505467,
                         0.11139509, 0.1098976 , 0.12266492, 0.22378939, 0.12720551])
#     @staticmethod
#     def mean():
#         return np.array([
#             0.00573914,  0.01986144, -0.15868089,  0.36768475,  0.72598995,  0.08102263,
#             0.25611847,  1.68464095,  0.19636668,  0.65685423, -0.14652498,  0.15565136,
#             -0.32924402,  1.6396836,  -0.31129081])

#     @staticmethod
#     def std():
#         return np.array([
#             0.12749956,  0.13707998,  0.13329148,  0.12998348,  0.14871537,  0.14387384,
#             0.16227234,  0.14832425,  0.16567627,  0.26399282,  0.12827283,  0.12381153,
#             0.14707848,  0.56716324,  0.15749387])

    @staticmethod
    def statistics(z):
        return z


class German(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/german/data.npy')
        labels = np.load('../data/german/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(German, self).__init__(data, labels, device)

    def mean(self):
        return np.array([-0.68741061,  0.38477283, -0.38415845,  0.13065674, -0.33949613,
                         -0.16838708, -0.14511187,  0.00994473,  0.17420383, -0.10977683,
                         -0.20910087,  0.10275232,  0.02527271, -0.12291449, -0.2590595 ,
                         0.25831413, -0.27745382,  0.24639396,  0.22202965,  0.10556248,
                         -0.07499404, -0.07947557, -0.01919004, -0.01645598, -1.17208187])
    
    def std(self):
        return np.array([0.08516181, 0.09635876, 0.08911405, 0.09956257, 0.08926239,
                         0.08701442, 0.07842944, 0.08611181, 0.0957857 , 0.09072278,
                         0.07546487, 0.08859214, 0.08167015, 0.08849675, 0.1060393 ,
                         0.07892459, 0.09604597, 0.10719325, 0.10019943, 0.11892524,
                         0.1235299 , 0.0847001 , 0.11354187, 0.11124831, 0.089358  ])
        
#     @staticmethod
#     def mean():
#         return np.array([
#             -0.73619639,  0.419458, -0.41486377,  0.12679717, -0.36520298, -0.1790139,
#             -0.15307771,  0.01321516,  0.18079792, - 0.11101034, - 0.22463548,  0.12258933,
#             0.02874339, -0.13638893, -0.29289896,  0.27896283, -0.29997425,  0.30485174,
#             0.27133239,  0.12250612, -0.06301813, -0.09286941, -0.02542205, -0.02292937,
#             -1.20507437])

#     @staticmethod
#     def std():
#         return np.array([
#             0.09370191,  0.1066482,   0.097784,    0.11055009,  0.09572253,  0.09415687,
#             0.08297686,  0.0928196,   0.10530122,  0.09953667,  0.07978824,  0.09610339,
#             0.0867488,   0.09550436,  0.11943925,  0.08431934,  0.10446487,  0.12292658,
#             0.11318609,  0.14040756,  0.1456459,   0.09062331,  0.13020753,  0.12852231,
#             0.09891565])

    @staticmethod
    def statistics(z):
        return z


class Heart(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/heart/data.npy')
        labels = np.load('../data/heart/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Heart, self).__init__(data, labels, device)
        
        
    def mean(self):
        return np.array([-0.00630095,  0.47727426,  0.5242346 ,  0.25273704,  0.21890952,
                         -0.15166965,  0.23752239, -0.36281532,  0.34540818,  0.36333619,
                         0.20744141,  0.72334647,  0.58762541, -0.27694406])
    
    def std(self):
        return np.array([0.1693578 , 0.17313657, 0.15845808, 0.15592132, 0.15881148,
                         0.15390558, 0.15432498, 0.17472474, 0.15798233, 0.18271763,
                         0.17119288, 0.17252138, 0.16023874, 0.17536533])

#     @staticmethod
#     def mean():
#         return np.array([
#             -0.13996868,  0.71390106,  0.69571619,  0.43944853,  0.36997702, -0.27319424,
#             0.31730518, -0.49617367,  0.40516419, 0.4312388,   0.26531786, 1.10337417,
#             0.70054367, -0.25684964])

#     @staticmethod
#     def std():
#         return np.array([
#             0.22915648,  0.24545612,  0.20457998,  0.20270157,  0.21040644,  0.20094482,
#             0.19749419,  0.24134014,  0.20230987,  0.25595334,  0.23709087,  0.24735325,
#             0.20701178,  0.19771984])

    @staticmethod
    def statistics(z):
        return z


class ICG:
    def __init__(self, device):
        self.num_dims = 50
        self.variances = torch.from_numpy(10**np.linspace(-2.0, 2.0, self.num_dims)).to(device).float()

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -0.5*torch.sum(x*(x*1/self.variances), dim=1)

    def mean(self):
        return np.zeros(self.num_dims)

    def std(self):
        return np.sqrt(10**np.linspace(-2.0, 2.0, self.num_dims))


class SCG:
    def __init__(self, device):
        self.num_dims = 2
        self.variances = torch.from_numpy(10 ** np.linspace(-2.0, 2.0, self.num_dims)).to(device).float()
        B = torch.from_numpy(np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)],
                                       [1 / np.sqrt(2), 1 / np.sqrt(2)]])).float().to(device)
        self.cov_matrix = B.mm(torch.diag(self.variances).mm(B.t()))
        self.inv_cov = torch.inverse(self.cov_matrix)

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -0.5*torch.sum(x*self.inv_cov.mm(x.t()).t(), dim=1)

    def mean(self):
        return np.zeros(2)

    def std(self):
        return torch.sqrt(torch.diag(self.cov_matrix)).cpu().numpy()


class L2HMC_MOGTwo:
    def __init__(self, device):
        self.num_dims = 2
        self.variance = 0.1
        self.mean1 = torch.Tensor([-2, 0]).to(device).float()
        self.mean2 = torch.Tensor([2, 0]).to(device).float()

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        prob1 = torch.exp(-0.5/self.variance*torch.sum((x - self.mean1) ** 2, dim=1))
        prob2 = torch.exp(-0.5/self.variance*torch.sum((x - self.mean2) ** 2, dim=1))
        return torch.log(prob1 + prob2 + 1e-10) - np.log(2.0)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    def std(self):
        return np.array([np.sqrt(4.1), np.sqrt(self.variance)])


class RoughWell:
    def __init__(self, device):
        self.num_dims = 2
        self.eta = 1e-2

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -torch.sum(0.5 * x*x + self.eta*torch.cos(x/self.eta), dim=1)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    def std(self):
        return np.ones(self.num_dims)
