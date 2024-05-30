import torch
import os
from .utils import *
from .bypass_bn import *

def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0
    ):
    loss = 0
    correct = 0
    total = 0
    
    if loop_type == 'train': 
        net.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            
            opt_name = type(optimizer).__name__
            
            if opt_name == 'WSAM':
                with torch.no_grad():
                    outputs = net(inputs)
                    first_loss = criterion(outputs, targets)
                optimizer.step_forward()
                enable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            elif opt_name == 'CHAUSAM3':
                with torch.no_grad():
                    outputs = net(inputs)
                    first_loss = criterion(outputs, targets)
                optimizer.step_backward()
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                criterion(net(inputs), targets).backward()
                optimizer.third_step(zero_grad=True)
            elif opt_name.startswith('CHAUSAM') or opt_name == 'OTHERSAM':
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                criterion(net(inputs), targets).backward()
                optimizer.third_step(zero_grad=True)
                if opt_name == 'CHAUSAM2':
                    criterion(net(inputs), targets).backward()
                    optimizer.forth_step(zero_grad=True)
            elif opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step(zero_grad=True)
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward(create_graph=True)        
                optimizer.first_step(zero_grad=True)
                # Zero the gradients explicitly
                for param in net.parameters():
                    param.grad = None
                
                if opt_name.startswith('VARSAM') or opt_name == "EXTRASAM":
                    disable_running_stats(net)  # <- this is the important line
                    second_loss = criterion(net(inputs), targets)
                    second_loss.backward(retain_graph=True)
                    optimizer.second_step(zero_grad=True)
                    
                    criterion(net(inputs), targets).backward()
                    optimizer.third_step(zero_grad=True)
                else:
                    disable_running_stats(net)  # <- this is the important line
                    criterion(net(inputs), targets).backward()
                    optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (loss_mean, acc, correct, total))
                
            try: 
                logging_dict[(f'{loop_type.title()}/hessian_norm', batch_idx)] = [optimizer.hessian_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/old_grad_norm', batch_idx)] = [optimizer.old_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/new_grad_norm', batch_idx)] = [optimizer.new_grad_norm, len(dataloader)]
            except: pass

            try: 
                logging_dict[(f'{loop_type.title()}/old_num_zero_grad', batch_idx)] = [optimizer.old_num_zero_grad, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/new_num_zero_grad', batch_idx)] = [optimizer.new_num_zero_grad, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/ratio_new_old_grad_norm', batch_idx)] = [optimizer.ratio_new_old_grad_norm, len(dataloader)]
            except: pass

            try: 
                logging_dict[(f'{loop_type.title()}/first_grad_norm', batch_idx)] = [optimizer.first_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/second_grad_norm', batch_idx)] = [optimizer.second_grad_norm, len(dataloader)]
            except: pass
           
            try: 
                logging_dict[(f'{loop_type.title()}/third_grad_norm', batch_idx)] = [optimizer.third_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/weight_norm', batch_idx)] = [optimizer.weight_norm, len(dataloader)]
            except: pass
          
            try: 
                logging_dict[(f'{loop_type.title()}/sim1', batch_idx)] = [optimizer.sim1, len(dataloader)]
            except: pass
    else:
        if loop_type == 'test':
            print('==> Resuming from best checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            load_path = os.path.join('checkpoint', logging_name, 'ckpt_best.pth')
            checkpoint = torch.load(load_path)
            net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (loss_mean, acc, correct, total))
        if loop_type == 'val':
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
            
            # if (epoch + 1) % 40 == 0:
            #     print(f'Saving {epoch+1}th checkpoint ...')
            #     state = {
            #         'net': net.state_dict(),
            #         'acc': acc,
            #         'loss': loss,
            #         'epoch': epoch
            #     }
            #     save_path = os.path.join('checkpoint', logging_name)
            #     if not os.path.isdir(save_path):
            #         os.makedirs(save_path)
            #     torch.save(state, os.path.join(save_path, f'{epoch+1}.pth'))
        
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'val': 
        return best_acc