from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/log')
f = open('log.txt', 'r')

history = {'train_loss':[], 'val_loss':[]}
lines = f.readlines()
for line in lines:
    loss = line.split('-')[1]
    if loss.split(':')[0] == " train loss":
        history['train_loss'].append(float(loss.split(':')[1]))
    else:
        history['val_loss'].append(float(loss.split(':')[1]))

epoch = []
for i in range(200):
    epoch.append(i)

for i in range(len(epoch)):
    writer.add_scalar('training loss', history['train_loss'][i], epoch[i])

for i in range(len(epoch)):
    writer.add_scalar('validation loss', history['val_loss'][i], epoch[i])