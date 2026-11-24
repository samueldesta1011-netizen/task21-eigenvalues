CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O2
LDLIBS = -lm

all: eigapp tests

eigapp: main.c eigen.c eigen.h
	$(CC) $(CFLAGS) main.c eigen.c $(LDLIBS) -o eigapp

tests: tests.c eigen.c eigen.h
	$(CC) $(CFLAGS) tests.c eigen.c $(LDLIBS) -o tests

test: tests
	./tests

run: eigapp
	./eigapp

clean:
	rm -f eigapp tests

.PHONY: all test run clean
