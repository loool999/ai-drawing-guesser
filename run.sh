#!/bin/bash

mkdir -p quickdraw_data
cd quickdraw_data

# List of classes to download
classes=(
    airplane alarm_clock ambulance angel animal_migration ant apple arm armchair asparagus axe backpack banana bandage barn baseball bat basket basketball bathtub beach bear beard bed bee belt bench bicycle binoculars bird birthday_cake blackberry blueberry book boomerang bottle_brush bottlecap bouquet bowtie bracelet brain bread bridge broccoli broom bucket bulldozer bus bush butterfly cactus cake calculator calendar camel camera camouflage campfire candle cannon canoe car car_battery car_race carrot castle cat ceiling_fan cello centaur cereal chair chandelier church circle clarinet clipboard clock cloud coffee_cup compass computer confetti cookie cooking_pot cooler couch cow crab crayon crocodile crow cruise_ship cup diamond dishwasher diving_board dog dolphin donut door dragon dresser drill drum duck dumbbell ear elbow elephant envelope eraser eye eyeglasses face fan feather fence fire_hydrant fire_truck fireplace fish flamingo flashlight flip_flops floor_lamp flower flying_saucer foot fork frog frying_pan garden gnome garden hose garden shed garden_shears giraffe glasses globe gloves goat grapes grass guitar hamburger hammer hand #harmonica harp hat headphones hedge hedgehog helicopter helmet hexagon hockey_puck hockey_stick horse hot_air_balloon hot_dog hot_tub hourglass house house_plant hula_hoop hurricane ice_cream ice_skate in_love ipod jam jar jeans jigsaw_puzzle jug kangaroo kayak keyboard knee koala ladder lantern laptop lasso leaf leg lightbulb lighthouse lightning line lion lobster lollipop mailbox map marker matches megaphone mermaid microphone microwave monkey moon mosquito motorbike mountain mouse mouth mug mushroom narwhal necklace nose octagon octopus owl paintbrush palette panda pants paper_clip parrot passport peach peanut peacock pear peas pencil penguin piano pickup_truck picture_frame pig pillow pineapple pizza plant pliers popsicle postcard postage_stamp power_outlet purse rabbit raccoon radio rainbow rake raspberry rhinoceros rifle river roller_skates roller_coaster rollers skate rocket roll_shoes rollerblades rope roundabout rubber_band running_man sailboat sandwich saw saxophone school_bus scissors scorpion screwdriver sea_seal seesaw shark sheep shield shoe shopping_cart shower shrimp skateboard skunk skull skyscraper sleeping_bag smiley_face snail snake snorkel snowboarding snowflake snowman soap sofa spaghetti speedboat spider spoon squirrel star steak steering_wheel stethoscope stitches stop_sign stork strawberry streetlight string_bean submarine suitcase sun sunglass_sun surfingman swing set sword swing set table teapot teddy_bear telephone telescope television tent thermometer tiger toilet toothbrush toothpaste tornado tractor traffic_light train transformer trash_can tree triangle trombone truck trumpet tulip turkey turtle umbrella van vase violin volcano volleyball washing_machine watermelon wheel wheelbarrow windmill window wine_bottle wine_glass wineglass wink wizard wolf wristwatch yarn zebra zigzag
)

# Download the files
for class in "${classes[@]}"; do
    file="${class}.npy"
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget -q "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${file}" -O "$file"
        if [ $? -ne 0 ]; then
            echo "Failed to download $file"
            rm -f "$file"
        fi
    fi
done

# Verify the integrity of each file
for file in *.npy; do
    python -c "import numpy as np; np.load('$file')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "File $file is corrupted, re-downloading..."
        wget -q "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${file}" -O "$file"
    fi
done
