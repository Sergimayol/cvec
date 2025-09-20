CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGET = cvec
SRC = cvec.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
