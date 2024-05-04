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
            
            if type(optimizer).__name__ == 'EXTRASAM':
                enable_running_stats(net)
                with torch.no_grad():
                    outputs = net(inputs)
                    first_loss = criterion(outputs, targets)
                optimizer.step_back()
                disable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.first_step(zero_grad=True)
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward(retain_graph=True)        
                optimizer.first_step(zero_grad=True)
                
                if type(optimizer).__name__.startswith('VARSAM'):
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
                logging_dict[(f'{loop_type.title()}/third_grad_norm', batch_idx)] = [optimizer.third_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/norm_d_norm_d_p', batch_idx)] = [optimizer.norm_d_norm_d_p, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/exp_avg_old_grad_norm_sq', batch_idx)] = [optimizer.exp_avg_old_grad_norm_sq, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/var_old_grad_norm_sq', batch_idx)] = [optimizer.var_old_grad_norm_sq, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/var_old_grad', batch_idx)] = [optimizer.var_old_grad, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/norm_exp_avg_d_norm_d_p', batch_idx)] = [optimizer.norm_exp_avg_d_norm_d_p, len(dataloader)]
            except: pass
              
            try: 
                logging_dict[(f'{loop_type.title()}/num_clamped_elements', batch_idx)] = [optimizer.num_clamped_elements, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/cnt_diff_sign', batch_idx)] = [optimizer.cnt_diff_sign, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/cnt_reg_diff_sign', batch_idx)] = [optimizer.cnt_reg_diff_sign, len(dataloader)]
            except: pass

            try: 
                logging_dict[(f'{loop_type.title()}/num_zero_elements', batch_idx)] = [optimizer.num_zero_elements, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/weight_norm', batch_idx)] = [optimizer.weight_norm, len(dataloader)]
            except: pass
          
            try: 
                logging_dict[(f'{loop_type.title()}/sim1', batch_idx)] = [optimizer.sim1, len(dataloader)]
                logging_dict[(f'{loop_type.title()}/sim2', batch_idx)] = [optimizer.sim2, len(dataloader)]
                logging_dict[(f'{loop_type.title()}/sim3', batch_idx)] = [optimizer.sim3, len(dataloader)]
                logging_dict[(f'{loop_type.title()}/sim4', batch_idx)] = [optimizer.sim4, len(dataloader)]
                logging_dict[(f'{loop_type.title()}/sim5', batch_idx)] = [optimizer.sim5, len(dataloader)]
            except: pass
            
            try: 
                name_list = [n for n, _ in net.named_parameters()]
                for group in optimizer.param_groups:
                    for n, p in zip(name_list, group["params"]):
                        logging_dict[(f'Summary/{n}_mean', batch_idx)] = [optimizer.state[p]['mean'], len(dataloader)]
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