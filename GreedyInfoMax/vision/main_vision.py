import torch
import time
import numpy as np
import gc

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.vision.data import get_dataloader


def validate(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):

        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss


def train(opt, model):
    # todo: remove
    total_step = len(train_loader)
    # total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    print("XXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXX")
    print(opt.num_epochs, opt.start_epoch)
    print("XXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXX")
    print("XXXXXXXXXXXXXXXX")

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for _ in range(opt.model_splits)]
        loss_updates = [1 for _ in range(opt.model_splits)]

      
        for step, (img, label) in enumerate(train_loader):
            
            # todo
            if step == 400:
                break
            # end todo

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0)  # Take mean over outputs of different GPUs.
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            model.zero_grad()
            overall_loss = sum(loss)
            overall_loss.backward()
            optimizer.step()

            for idx, cur_losses in enumerate(loss):
                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

        if opt.validate:
            validation_loss = validate(opt, model, test_loader)  # Test_loader corresponds to validation set here.
            logs.append_val_loss(validation_loss)

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":


    # added myself
    torch.cuda.empty_cache()
    gc.collect()


    # --grayscale --download_dataset --save_dir vision_experiment

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    print("XXXXXXXXXXXXXXXXXXX")
    print(opt)
    print(opt.batch_size_multiGPU)
    print("XXXXXXXXXXXXXXXXXXX")

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)
