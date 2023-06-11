import sys
import os

import torch
import torchvision as tv
from torchvision.io import read_image

from transformer import resizeImage, addGaussianNoise, HWRTransformer

# input height, input width, nr of chars, and longest label
#   hardcoded here as it is dependent on original dataset
IMG_WIDTH = 2260
IMG_HEIGHT = 128
NR_OF_TOKENS = 82
LONGEST_LABEL = 56
MEAN = 237.3417
STD = 43.9404



def main(input_dir_path):
    char_to_idx_mapping = {'<PAD>': 0,'<BOS>': 1,'<EOS>': 2,' ': 3,'e': 4,'t': 5,'a': 6,'o': 7,
        'n': 8,'i': 9,'s': 10,'r': 11,'h': 12,'l': 13,'d': 14,'c': 15,'u': 16,'m': 17,'f': 18,
        'p': 19,'w': 20,'g': 21,'y': 22,'b': 23,'.': 24,',': 25,'v': 26,'k': 27,"'": 28,'"': 29,
        '-': 30,'T': 31,'I': 32,'M': 33,'A': 34,'S': 35,'B': 36,'P': 37,'H': 38,'W': 39,'C': 40,
        'N': 41,'G': 42,'x': 43,'R': 44,'L': 45,'E': 46,'D': 47,'F': 48,'0': 49,'1': 50,'j': 51,
        'O': 52,'q': 53,'!': 54,'U': 55,'(': 56,'K': 57,'?': 58,'z': 59,'3': 60,')': 61,'9': 62,
        ';': 63,'V': 64,'2': 65,'J': 66,'Y': 67,':': 68,'5': 69,'8': 70,'4': 71,'6': 72,'#': 73,
        '&': 74,'7': 75,'/': 76,'Q': 77,'X': 78,'*': 79,'Z': 80,'+': 81
        }

    idx_to_char_mapping = {value: key for key, value in char_to_idx_mapping.items()}

    device = "cpu"

    transforms = tv.transforms.Compose([resizeImage(IMG_WIDTH, IMG_HEIGHT),
                                        tv.transforms.Normalize(MEAN, STD),
                                        # addGaussianNoise()])
                                        ])

    # load model
    transformer = HWRTransformer(IMG_HEIGHT, IMG_WIDTH, NR_OF_TOKENS, LONGEST_LABEL)

    # load model weights
    transformer.load_state_dict(torch.load("model.pth", map_location = torch.device("cpu")))

    # make ouput folder
    if not os.path.exists("results"):
        os.mkdir("results")

    for file in os.listdir(input_dir_path):
        if os.path.isfile(os.path.join(input_dir_path, file)):
            input_img = read_image(os.path.join(input_dir_path, file)).float()

            # prepare image as input
            transformed_img = transforms(input_img).unsqueeze(0)

            # init decoder input (init -> 0 = <PAD>)
            decoder_in = torch.zeros(LONGEST_LABEL)
            # insert first <BOS> token
            decoder_in[0] = char_to_idx_mapping["<BOS>"]
            decoder_in = decoder_in.unsqueeze(0).int()
            # print(decoder_in.shape)


            output_string = ""

            with torch.no_grad():
                transformer.eval()
                for i in range(1, 56):
                    _, output = transformer(transformed_img, decoder_in)
                    # output idx for current step
                    # print(output[i][0])
                    # print(output[i][0][2:])
                    current_char_idx = torch.argmax(output[i][0][2:]).item() + 2
                    char = idx_to_char_mapping[current_char_idx]
                    if char == '<EOS>':
                        # print("<EOS> token predicted")
                        break
                    
                    # add prediction to input
                    decoder_in[0][i] = current_char_idx

                    # add to output string
                    # print(char)
                    output_string += char
            
            print(output_string)
            file = open("results/" + file[:-4] + "_characters.txt", "w")
            file.write(output_string)

if __name__ == '__main__':
    # main(sys.argv[0])
    main(sys.argv[1])
