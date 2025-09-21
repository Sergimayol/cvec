CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp -I.
TARGET = main
SRC = main.c
TEST = tests/tests

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

$(TEST): tests/tests.c
	$(CC) $(CFLAGS) -o $(TEST) tests/tests.c -lm

run: $(TARGET)
	./$(TARGET)

test: $(TEST)
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST)
