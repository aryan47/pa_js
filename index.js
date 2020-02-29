const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const natural = require('natural')
const tokenizer = require('@tensorflow-models/universal-sentence-encoder')
const readline = require('readline-sync')
natural.LancasterStemmer.attach();

const files = fs.readdirSync("./dataset")
const model = tf.sequential();

var embed;
var train_x = [],
    train_y = [];


// stem and tokenize text
const textPreProcess = (text) => {
    return text.tokenizeAndStem();
}

// file preprocess
const filePreprocess = (f_content) => {
    f_content = JSON.parse(f_content);
    f_content.train_data.data.forEach((eachData) => {
        train_x.push(textPreProcess(eachData.text).join(" "));
        train_y.push(textPreProcess(eachData.intent).toString());
    })

}

// iterate through all the files
files.forEach((file) => {
    let f_content = fs.readFileSync(__dirname + "/dataset/" + file, 'utf-8');
    filePreprocess(f_content);
})

const embeddingLayer = async () => {

    let sentence = train_x;
    embed = await tokenizer.load();

    return embed.embed(sentence);
}

var oneHotEncode = function (data) {
    let unique = [...new Set(data)];
    let encodedData = []
    data.forEach((value) => {
        let encoded = tf.zeros([unique.length], "int32").dataSync();
        encoded[unique.indexOf(value)] = 1;
        encodedData.push(encoded.slice());
    })
    return tf.tensor2d(encodedData);

}
var chat = async () => {
    const user = readline.question("You: ");
    const data = textPreProcess(user).join(" ");

    await embed.embed(data).then(data => {

        console.log(`data: ${data} `);
        const predicted = model.predict(data);
        console.log(`predicted: ${predicted.dataSync().indexOf(...predicted.max().dataSync())} `);
        return predicted.dataSync().indexOf(...predicted.max().dataSync())
    });


}
embeddingLayer().then(async data => {
    console.log(data);
    train_x = data;
    train_y = oneHotEncode(train_y);
    model.add(tf.layers.dense({
        inputShape: 512,
        activation: "softmax",
        units: 2
    }))

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.001),
        metrics: ["accuracy"]
    })
    await model.fit(train_x, train_y, { epochs: 200 });

 
    while (true) {

        await chat();
    }
})




