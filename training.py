for epoch in range(epochs):
    for batch in dataset:
        train_discriminator(batch)
        train_generator(batch)
// saving the trained model

generator.save('image_translation_generator.h5')

discriminator.save('image_translation_discriminator.h5')

