#!/bin/bash

compile() {
    make clean
    make
}

compile_test() {
    make compile_test
}

install_lib() {
    cp -r include/mole_math /usr/local/include/ 
    compile
    cp lib/libmolemath.so /usr/local/lib/ 
    ldconfig
    compile_test
}

case $1 in
    install)
        install_lib
        exit 0
        ;;
    uninstall)
        rm /usr/local/lib/libmolemath.so
        rm -r /usr/local/include/mole_math/
        exit 0
        ;;
    *)
        if [ -z "$1" ]
        then
            install_lib
            exit 0
        fi
        echo "Error: Unrecognized command."
        exit 0
        ;;
esac