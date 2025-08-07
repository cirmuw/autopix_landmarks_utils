from pprint import pprint

import landmarks_utils
import landmarks_utils.utils.utils

# import nnunetv2
# from pprint import pprint


def main():
    package_info_parameters = {
        "landmarks_utils": landmarks_utils.utils.utils.package_infos(landmarks_utils),
        #"nnunetv2": cseg_utils.utils.package_infos(nnunetv2)
    }
    print("package_infos:::")
    pprint(package_info_parameters)

if __name__ == "__main__":
    main()
