import os
import torch
import numpy as np
import cv2
colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
          (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26)]

class Infer():
    '''
    Class for main inference
    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["local"]["common_size"] = 1024;
        self.system_dict["local"]["mean"] = np.array([[[0.485, 0.456, 0.406]]])
        self.system_dict["local"]["std"] = np.array([[[0.229, 0.224, 0.225]]])

    def Model(self, model_dir="trained/"):
        '''
        User function: Selet trained model params
        Args:
            model_dir (str): Relative path to directory containing trained models 
        Returns:
            None
        '''
        self.system_dict["local"]["model"] = torch.load(model_dir + "/signatrix_efficientdet_coco.pth").module
        if torch.cuda.is_available():
            self.system_dict["local"]["model"] = self.system_dict["local"]["model"].cuda();

    def Predict(self, img_path, class_list, vis_threshold = 0.4, output_folder = 'Inference'):
        '''
        User function: Run inference on image and visualize it
        Args:
            img_path (str): Relative path to the image file
            class_list (list): List of classes in the training set
            vis_threshold (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            output_folder (str): Path to folder where output images will be saved
        Returns:
            tuple: Contaning label IDs, Scores and bounding box locations of predicted objects. 
        '''
        try:
          if not os.path.exists(output_folder):
            os.makedirs(output_folder)
          # image path for image which we wnat to predict it
          image_filename = os.path.basename(img_path)
          # read that img
          img = cv2.imread(img_path);
          # imgae are in BGR color by used cv2 and to get high performance in model we convert it to RGB
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
          image = img.astype(np.float32) / 255.;
          # do Normlizer
          image = (image.astype(np.float32) - self.system_dict["local"]["mean"]) / self.system_dict["local"]["std"]
          # fine Resoluation (Shape) for image
          height, width, _ = image.shape
          # if image size not as common size to resize image
          if height > width: 
              scale = self.system_dict["local"]["common_size"] / height
              resized_height = self.system_dict["local"]["common_size"]
              resized_width = int(width * scale)
          else:
              scale = self.system_dict["local"]["common_size"] / width
              resized_height = int(height * scale)
              resized_width = self.system_dict["local"]["common_size"]
          # scale image resize to fit with common size
          image = cv2.resize(image, (resized_width, resized_height))
          # create initi image to assign our image with same size 
          #print("W * H :", resized_width, " * ", resized_height)
          new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
          new_image[0:resized_height, 0:resized_width] = image

          img = torch.from_numpy(new_image)

          with torch.no_grad():
            # send our image to our model to predict 
            # permut image to tensor permut(input,Dimension)>>>tesnor
              scores, labels, boxes = self.system_dict["local"]["model"](img.cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
              # print("scores: ",scores)
              # print("labels: ",labels)
              
              boxes /= scale;
              #print("boxes: ",boxes)
        except Exception as e:
          print(e)


        try:

            isDetect=False
            #print("boxes: ",boxes)
            if boxes.shape[0] > 0:
                isDetect=True
                icount=0
                #predCount=0
                output_image = cv2.imread(img_path)
                #print(img_path, "Predected ")
                for box_id in range(boxes.shape[0]):
                    #pdb.set_trace()
                    pred_prob = float(scores[box_id])
                    #icount+=1
                    #print(" for this image : ", img_path)
                    #print("pred_prob  before check Vis_Threshold:",pred_prob)
                    if pred_prob < vis_threshold:
                        #print("pred_prob < 0.5 (Thresold): ", pred_prob)
                        
                        break
                    #print("pred_prob greatThan vis_Threshold:",pred_prob)
                    #predCount+=1
                    icount+=1
                    pred_label = int(labels[box_id])
                    #print("pred_label :",pred_label)
                    xmin, ymin, xmax, ymax = boxes[box_id, :]
                    color = colors[pred_label]
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = cv2.getTextSize(class_list[pred_label] + ' : %.2f' % pred_prob, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    #print("text_size : ",text_size)
                    cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                    cv2.putText(
                        output_image, class_list[pred_label] + ' : %.2f' % pred_prob,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

                    cv2.imwrite(os.path.join(output_folder, image_filename), output_image)
                    cv2.imwrite("output.png", output_image)
                # save reslut of predictions in txt files 
                scores = scores.tolist()
                boxes = boxes.tolist()
                labels = labels.tolist()
                imageName=image_filename.split(".jpg",1)[0] # in Drone test image extenstion is JPEG not jpg or JPG
                #print("output_folder :",output_folder)
                #print("image_filename :",image_filename)
                # we save prediction result in voc format in txt files
                with open(output_folder +"/"+ imageName + '.txt', 'w') as out_file:# txt file contains result of inference img result
                  #write detection result
                  for i in range(len(scores)):
                    #print(scores)
                    line = class_list[labels[i]] + ' ' + str(scores[i]) + ' ' + str(' '.join([str(j) for j in boxes[i]]))
                    out_file.write(line)
                    if i < len(scores) - 1:
                      out_file.write('\n')
              
            return scores, labels, boxes,isDetect,icount

        except Exception as e:
            print("NO Object Detected in image: ",image_filename)
            print("error : ",e)

            return None



        