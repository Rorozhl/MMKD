cifar100_teacher_model_name = [

    'cifar100-vgg13-0', 'cifar100-vgg13-1', 'cifar100-vgg13-2',
    # 'cifar100-resnet56-0', 'cifar100-resnet20x4-1', 'cifar100-vgg13-2', 

]


dogs_teacher_model_name = [
    # 'dogs-ResNet34x2-0', 'dogs-ResNet34x2-1', 'dogs-ResNet34x2-2',
    'dogs-ResNet34x4-0', 'dogs-ResNet34x4-1', 'dogs-ResNet34x4-2',
    # 'dogs-ResNet101-0', 'dogs-ResNet101-1', 'dogs-ResNet101-2',
    # 'dogs-ResNet50-0', 'dogs-ResNet50-1', 'dogs-ResNet50-2',
]

tinyimagenet_teacher_model_name = [
    'tinyimagenet-resnet32x4-0', 'tinyimagenet-resnet32x4-1', 'tinyimagenet-resnet32x4-2',
    # 'tinyimagenet-vgg13-0', 'tinyimagenet-vgg13-1', 'tinyimagenet-vgg13-2',
]


cifar100_student_model_name = [

]


# ------------- teacher net --------------------#
teacher_model_path_dict = {

    'cifar100-vgg13-0': '../save/MultiT/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_1/vgg13_best.pth',
    'cifar100-vgg13-1': '../save/MultiT/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_2/vgg13_best.pth',
    'cifar100-vgg13-2': '../save/MultiT/teachers/models/vgg13_cifar100_lr_0.05_decay_0.0005_trial_4/vgg13_best.pth', 
}

# ------------- student net --------------------#

student_model_path_dict = {
    
}