# Stock Analyst
The convenience assistant to understanding stock prices cached into an SQL database!
<img src = "images/llamaindexsql.png">

### Setup
Install requirements
```poetry install```

Create a .env file and add ElevenLabs API key (for text to accented speech) and OpenAI API key to it.

Run `notebooks>2.agent_stock.ipynb` to get stock data from Yahoo Finance and store it to a local SQL database.

Change images in the `public` file to reflect your app's logo and favicon

Setup chainlit server
```chainlit run app.py```

Add the following lines of code to the body tag of your HTML file
```
<body>
    <script src="{chainlit deployment URL here}copilot/index.js"></script>
    <script> 
        window.mountChainlitWidget({
        chainlitServer: "{chainlit deployment URL here}",
        });
        window.addEventListener("chainlit-call-fn", (e) => {
            const { name, args, callback } = e.detail;
            if (name === "test") {
              callback(args.response);
            }
          });
    </script>
</body>
```