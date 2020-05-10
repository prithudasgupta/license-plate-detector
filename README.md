license-plate-detector

Using feature extraction and CNNs, this project reads in potential images of cars, locates this license plate on the car, and reads the license plate number.

To run:
1. Navigate to the code directory
2. Run the command python3 main.py --generate-weights which will train the model from scratch and save model weights to the saved_weights folder.
3. Test model using the --load-weight and --test-uploaded-image flags. A few examples that can be run are:

    python3 main.py --load-weight <path_to_weight_in_saved_weights_dir> --test-uploaded-image data_license_only/crop_h1/I00016.png
    ^expected output: 4B27433

    python3 main.py --load-weight <path_to_weight_in_saved_weights_dir> --test-uploaded-image data_license_only/crop_h2/I00055.png
    ^expected output: 1BA0888

    python3 main.py --load-weight <path_to_weight_in_saved_weights_dir> --test-uploaded-image data_license_only/crop_h4/I00011.png
    ^expected output: 3M46918
