I used the following steps to set up a sample dataset for testing purposes:

1. **Importing Event Reactions:**
   I imported the `data/logs_*.csv` files into the `eventReactions` collection using this `mongoimport` command via `mongosh`:
   
   ```bash
   mongoimport --uri "mongodb://myusername:mypassword@myhost/mydb" --collection eventReactions --type csv --headerline --file path/to/logs.csv
   ```

2. **Creating User Records:**
   Next, I generated user records using the Users schema from the chorus project. I made a slight alteration by enhancing data denormalization and adding an events array to each user. It's worth noting that denormalizing data is a recommended practice in NoSQL database systems like MongoDB. This process was performed using a `js` script named `populate_users.js`, which I executed within the `mongosh` environment.

3. **Generating Event Data:**
   To further enhance testing, I employed the `populate_events` script to replicate a basic dataset containing event information. This dataset proved valuable for the development and testing of my code.