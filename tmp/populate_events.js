// Define event data
const events = [
    {
      owner: '64e06940888549c5a4aec13a',
      title: 'Test 1',
      uuid: '64650ec61ea2ae40b211ccbb'
    },
    {
      owner: '64e06940888549c5a4aec13a',
      title: 'Test 2',
      uuid: '645a47551ea2ae40b211ca64'
    }
  ];
  
  // Insert event documents into the 'events' collection
  db.events.insertMany(events);
  
  print(`${events.length} events inserted successfully.`);
  