from mxnet import nd, gluon, autograd
import mxnet as mx
from mxnet.gluon.loss import _reshape_like
import numpy as np
import net_sphere
def transform(data,label):
    data = data.astype(np.float32)-127.5
    data = data * 0.0078125
    return nd.transpose(data, (2, 0, 1)), float(label)


batch_size=8
train_set = mx.gluon.data.vision.ImageRecordDataset("/home/liuhao/dataset/VGGFACE/vggface64.rec",transform=transform)
train_loader = mx.gluon.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,last_batch="keep")

ctx=[mx.gpu(2)]

my_net = net_sphere.sphere_net(classnum=8528,m=1)

#mobilenet = gluon.model_zoo.vision.mobilenet1_0(pretrained=False,ctx=ctx,classes=10)

my_net.initialize(init=mx.init.Xavier(),ctx=ctx)

sphere_loss=net_sphere.AngleLoss()

trainer=gluon.Trainer(my_net.collect_params(),'adam',{'learning_rate':0.5})

epoches=30
for e in range(epoches):
    mean_loss = 0
    my_net.get_feature = False
    for i, (data,label) in enumerate(train_loader):
        data_list=gluon.utils.split_and_load(data,ctx)
        label_list=gluon.utils.split_and_load(label,ctx)

        with autograd.record():#training mode
            losses=[sphere_loss(my_net(X),y) for X,y in zip(data_list,label_list)]

        for l in losses:
            l.backward()
        trainer.step(batch_size=data.shape[0])

        mean_loss+=(np.sum([l.mean().asscalar() for l in losses])/data.shape[0])
        print("Epoch %s. Batch %s. Loss: %s"(e, i, mean_loss/(i+1)))

    nd.waitall()

    print("Epoch %s. Loss: %s" % (e, mean_loss/(i+1)))