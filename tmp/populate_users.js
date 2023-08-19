const crypto = require('crypto');

// Hash the password using MD5 (for demonstration purposes)
const hashPassword = password => crypto.createHash('md5').update(password).digest('hex');

// Define user data
const users = [
  {
    username: 'mostafa',
    password: hashPassword('password'), // Replace with the actual password
    role: 'ADMIN'
  },
  {
    username: 'mahdi',
    password: hashPassword('password'), // Replace with the actual password
    events: ['64650ec61ea2ae40b211ccbb', '645a47551ea2ae40b211ca64']
  }
];

// Insert user documents into the 'user' collection
users.forEach(user => {
  db.user.insertOne(user);
});

print(`${users.length} users inserted successfully.`);
