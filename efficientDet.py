import torch.nn as nn
import torch
import math
from model import BiFPN,Regressor,Classifier,EfficientNet
from utils import BBoxTransform, ClipBoxes, Anchors
from loss import FocalLoss
from torchvision.ops.boxes import nms as nms_torch


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)
    
class EfficientDet(nn.Module):
    def __init__(self, num_anchors=9, num_classes=20, compound_coef=0, model_name="efficientnet-b0"):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef

        self.num_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        ''' why we start from level 3 conv3 to conv7 (beased on Paper "These fused features are fed to a class and box
          network to produce object class and bounding box predictions
          respectively.")'''
        if(self.compound_coef == 0 or self.compound_coef==1):
          self.conv3 = nn.Conv2d(40, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(80, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(192, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(192, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 2):
          self.conv3 = nn.Conv2d(48, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(88, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(208, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(208, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 3):
          self.conv3 = nn.Conv2d(48, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(96, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(232, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(232, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 4):
          self.conv3 = nn.Conv2d(56, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(112, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(272, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(272, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 5):
          self.conv3 = nn.Conv2d(64, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(128, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(304, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(304, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 6):
          self.conv3 = nn.Conv2d(72, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(144, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(344, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(344, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        elif(self.compound_coef == 7):
          self.conv3 = nn.Conv2d(80, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv4 = nn.Conv2d(160, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv5 = nn.Conv2d(384, self.num_channels, kernel_size=1, stride=1, padding=0)
          self.conv6 = nn.Conv2d(384, self.num_channels, kernel_size=3, stride=2, padding=1)
          self.conv7 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))
        



        self.bifpn = nn.Sequential(*[BiFPN(self.num_channels) for _ in range(min(2 + self.compound_coef, 8))])

        ''' To understand nn.Sequential
        model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
         output of model will be

         Sequential(
        (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
        (3): ReLU()
      )

        '''

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.num_channels, num_anchors=num_anchors,
                                   num_layers=3 + self.compound_coef // 3)
        self.classifier = Classifier(in_channels=self.num_channels, num_anchors=num_anchors, num_classes=num_classes,
                                     num_layers=3 + self.compound_coef // 3)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

        self.backbone_net = EfficientNet(model_name)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        if len(inputs) == 2:
            is_training = True
            img_batch, annotations = inputs
        else:
            is_training = False
            img_batch = inputs

        c3, c4, c5 = self.backbone_net(img_batch)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        features = [p3, p4, p5, p6, p7]
        features = self.bifpn(features)

        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        if is_training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

