#!/bin/bash

compile() {
    make clean
    make
}

copy_to_usr_lib() {
    cp lib/libmolemath.so /usr/local/lib/ 
	cp -r include/mole_math /usr/local/include/ 
}

case $1 in
    uninstall)
        rm /usr/local/lib/libmolemath.so
        rm -r /usr/local/include/mole_math/
        exit 0
        ;;
    *)
        echo "Error: Unrecognized command."
        exit 0
        ;;
esac

compile
copy_to_usr_lib
ldconfig