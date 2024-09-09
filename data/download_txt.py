import os.path

import requests


if __name__ == '__main__':
    input_list = '/home/mil/gourdin/inr_3d_data/data/vertebra_medshapenet.txt'
    output_dir = '/home/mil/gourdin/inr_3d_data/data/medshapenet_vertebra/'
    file = open(input_list, 'r')
    data = file.read()
    data_list = data.split("\n")
    file.close()
    for url in data_list:
        r = requests.get(url, allow_redirects=True)
        file_name = url.split('=')[-1]
        new_path = os.path.join(output_dir, file_name)
        open(new_path, 'wb').write(r.content)
        print("downloaded: " + str(file_name))
