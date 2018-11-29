#model predictor
example = input('Test: ')
example = sentence_to_wordlist(example)
example = tokenizer.texts_to_sequences([example])
example = pad_sequences(example, maxlen=maxlen)

category_probability = model.predict(example)

print(max(category_probability[0]))

number = np.argmax(category_probability)
if number == 1:
    print('Society & Culture')
elif number == 2:
    print('Science & Math')
elif number == 3:
    print('Health')
