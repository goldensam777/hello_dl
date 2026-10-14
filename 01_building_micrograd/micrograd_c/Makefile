CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -I.
LDFLAGS = -lm

TARGET = micrograd
SOURCES = main.c micrograd.c topological_sort/topo.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.c micrograd.h
	$(CC) $(CFLAGS) -c $< -o $@

topological_sort/topo.o: topological_sort/topo.c topological_sort/topo.h micrograd.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
