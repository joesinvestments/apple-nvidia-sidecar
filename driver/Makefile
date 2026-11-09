# libtgd + tgdprobe + test. Plain user-space binaries, unsigned: no entitlements are needed
# when the dext carries allow-any-userclient-access.
CFLAGS = -O2 -Wall -Wextra -arch arm64

all: libtgd.dylib tgdprobe test_tgd

libtgd.dylib: libtgd.c libtgd.h
	clang $(CFLAGS) -dynamiclib -install_name @rpath/libtgd.dylib -framework IOKit -framework CoreFoundation -o $@ libtgd.c

tgdprobe: tgdprobe.c libtgd.dylib
	clang $(CFLAGS) -o $@ tgdprobe.c -L. -ltgd -Wl,-rpath,@executable_path

test_tgd: test_tgd.c libtgd.dylib
	clang $(CFLAGS) -o $@ test_tgd.c -L. -ltgd -Wl,-rpath,@executable_path

clean:
	rm -f libtgd.dylib tgdprobe test_tgd
