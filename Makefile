CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp
TARGET = main
SRC = main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
