from mxnet import nd, gluon
import mxnet as mx
from mxnet.gluon.loss import _reshape_like
import numpy as np

class BasicBlock1(gluon.nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(BasicBlock1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels,kernel_size=3, strides=2, padding=1)
            self.prelu1 = gluon.nn.LeakyReLU(alpha=0.25)
            self.conv2 = gluon.nn.Conv2D(channels, kernel_size=3, strides=1, padding=1)
            self.prelu2 = gluon.nn.LeakyReLU(alpha=0.25)
            self.conv3 = gluon.nn.Conv2D(channels, kernel_size=3,strides=1,padding=1)
            self.prelu3 = gluon.nn.LeakyReLU(alpha=0.25)

    def hybrid_forward(self, F, x):
        x=self.prelu1(self.conv1(x))
        return x+self.prelu3(self.conv3(self.prelu2(self.conv2(x))))


class BasicBlock2(gluon.nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(BasicBlock2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels, kernel_size=3, strides=1, padding=1)
            self.prelu1 = gluon.nn.LeakyReLU(alpha=0.25)
            self.conv2 = gluon.nn.Conv2D(channels, kernel_size=3, strides=1, padding=1)
            self.prelu2 = gluon.nn.LeakyReLU(alpha=0.25)

    def hybrid_forward(self, F,x):
        x = self.prelu1(self.conv1(x))
        return x + self.prelu2(self.conv2(x))

class AngleLoss(gluon.loss.Loss):
    def __init__(self, batch_axis=0,**kwargs):
        super(AngleLoss, self).__init__(None,batch_axis=batch_axis, **kwargs)
        self.it=0
        self.LambdaMin=5.0
        self.LambdaMax=1500.0
        self.lamb=1500.0

    def hybrid_forward(self, F, x, label):
        self.it+=1
        cos_theta, phi_theta=x
        bathch_size=cos_theta.shape[0]
        label = label.reshape((-1, 1))
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output=cos_theta*1.0#f_y_i

        index = np.arange(0, bathch_size).reshape((-1, 1)).astype(np.int)
        dis = np.zeros(shape=cos_theta.shape, dtype=np.float32)
        labe_np = label.asnumpy().astype(np.int)

        cos_theta_np = cos_theta.asnumpy()
        phi_theta_np = phi_theta.asnumpy()
        dis[index,labe_np]-=cos_theta_np[index,labe_np]*(1.0)/(1+self.lamb)
        dis[index,labe_np]+=phi_theta_np[index,labe_np]*(1.0)/(1+self.lamb)#(lamda*cos+theta)/(1+lamda)
        dis_nd = F.array(dis)
        output_m = (output + dis_nd)
        # temp = output[index,label]
        # output[index,label]=nd.where(temp>(-self.m),
        #                              temp+self.m,temp)
        # output = output.asnumpy()
        # label_one_hot = label_one_hot.asnumpy().astype(np.int)
        # output[label_one_hot]=np.where(output[label_one_hot] > (-self.m),output[label_one_hot]+self.m,output[label_one_hot])
        # output[np.where(output[label_one_hot] < (-self.m))] += (-self.m)

        loss = F.softmax_cross_entropy(output_m, label[:, 0].astype(np.float32))

        #
        # logpt=F.log_softmax(nd.array(output))
        # label = label.astype(np.int)
        # index = F.concat(nd.arange(bathch_size).reshape((1,-1)).astype(np.int),label,dim=0)
        # logpt=F.gather_nd(logpt,index)
        # logpt=logpt.reshape((1,-1))
        #
        # pt=F.exp(logpt)
        # loss=-1*(1-pt)**self.gamma*logpt
        # loss=F.mean(loss)
        return loss

class AngleLinear(gluon.nn.HybridBlock):
    def __init__(self, unit, in_unit=0, m=4):
        super(AngleLinear, self).__init__()
        with self.name_scope():
            self.in_unit=in_unit
            self.unit=unit
            self.m=m
            #\cos(m\theta_i)=\sum_n(-1)^n{C_m^{2n}\cos^{m-2n}(\theta_i)\cdot(1-\cos(\theta_i)^2)^n}, (2n\leq m)
            self.mlambda = [
                lambda x: x ** 0,
                lambda x: x ** 1,
                lambda x: 2 * x ** 2 - 1,
                lambda x: 4 * x ** 3 - 3 * x,
                lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
                lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
            ]
            self.cnt=0

    def hybrid_forward(self, F, x, weight):#x(B,in)
        if self.cnt==0:
            self.in_unit=x.shape[1]
            with self.name_scope():
                self.weight = self.params.get('weight', shape=(self.in_unit, self.unit), allow_deferred_init=True)
                self.weight.initialize(mx.init.Xavier(magnitude=2.24))
            self.cnt+=1

        # norm weight
        w = weight
        # w = self.weight.data()#w(in,out)
        w_norm = F.sqrt(F.sum(F.square(w + 1e-6), axis=0)).reshape((1, self.unit))
        x_norm = F.sqrt(F.sum(F.square(x + 1e-6), axis=1)).reshape((-1, 1))

        w_normalize = F.broadcast_div(w, w_norm)
        # w_normalize = w/(w_norm)
        # w_norm = nd.sqrt(nd.sum(nd.square(w_normalize), axis=0)).reshape((1, self.unit))
        # orignal ouput
        output = F.dot(x, w_normalize)  # (B,out)

        cos_theta = F.broadcast_div(output, x_norm)
        # cos_theta=output/(x_norm)
        cos_theta = F.clip(cos_theta, -1, 1)

        cos_m_theta=self.mlambda[self.m](cos_theta)
        theta = F.arccos(cos_theta)
        k=F.floor(self.m*theta/np.pi)
        n_one=-1.0
        phi_theta=(n_one**k)*cos_m_theta-2*k

        cos_theta = F.broadcast_mul(cos_theta,x_norm)
        phi_theta = F.broadcast_mul(phi_theta,x_norm)
        # cos_theta=cos_theta*x_norm
        # phi_theta=phi_theta*x_norm

        nd_stack = F.stack(cos_theta,phi_theta)

        return nd_stack

class sphere_net(gluon.nn.HybridBlock):
    def __init__(self, classnum = 10574, feature = False, m = 4, **kwargs):
        super(sphere_net, self).__init__()
        self.classnum=classnum
        self.feature=feature
        self.m=m
        self.net_sequence = gluon.nn.HybridSequential()
        self.get_feature = False
        #input B*3*112*96
        with self.net_sequence.name_scope():
            self.net_sequence.add(BasicBlock1(channels=64),
                                  BasicBlock1(channels=128),
                                  BasicBlock2(channels=128),
                                  BasicBlock1(channels=256),
                                  BasicBlock2(channels=256),
                                  BasicBlock2(channels=256),
                                  BasicBlock2(channels=256),
                                  BasicBlock1(channels=512))

            self.net_sequence.add(gluon.nn.Dense(512))

            self.AngleLinear = AngleLinear(self.classnum,m=self.m)
            #self.net_sequence.add(AngleLinear(self.classnum,m=self.m))
    def hybrid_forward(self, F, x):
        if self.get_feature:
            return self.net_sequence(x)
        else:
            return self.AngleLinear(self.net_sequence(x))
