import React, { useEffect, useState } from 'react';
import { 
  StreamVideo, 
  StreamVideoClient, 
  StreamCall, 
  SpeakerLayout, 
  CallControls, 
  StreamTheme 
} from '@stream-io/video-react-sdk';

import '@stream-io/video-react-sdk/dist/css/styles.css';
import MyParticipantList from './Participants';

const apiKey = import.meta.env.VITE_STREAM_API_KEY; // Ensure this matches your backend .env
const callId = 'session101';

const App = () => {
  const [client, setClient] = useState();
  const [call, setCall] = useState();

  // 1. Logic to keep the same "random" ID across refreshes
  const [userId] = useState(() => {
    const savedId = localStorage.getItem('stream_user_id');
    if (savedId) return savedId;
    
    // Generate new random ID: e.g., "user_7n3k2z"
    const newId = `user_${Math.random().toString(36).substring(2, 8)}`;
    localStorage.setItem('stream_user_id', newId);
    return newId;
  });

  useEffect(() => {
    const initClient = async () => {
      try {
        // 2. Fetch the token from your FastAPI /token endpoint
        console.log(userId)
        const response = await fetch(`http://127.0.0.1:8000/token/${userId}`);
        const { token } = await response.json();

        const _client = new StreamVideoClient({
          apiKey,
          user: { id: userId, name: `Player-${userId.split('_')[1]}` },
          token,
        });

        setClient(_client);
      } catch (err) {
        console.error("Auth failed. Check if FastAPI is running on port 8000", err);
      }
    };

    initClient();
    return () => client?.disconnectUser(); // Cleanup [cite: 215, 229]
  }, [userId]);

  useEffect(() => {
    if (!client) return;

    const setupCall = async () => {
      const _call = client.call('default', callId);
      await _call.join({ create: true }); // [cite: 213]
      setCall(_call);

      // 3. Trigger the Vision Agent session on your backend
      await fetch('http://127.0.0.1:8000/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_id: callId, call_type: 'default' }),
      });
    };

    setupCall();
    return () => call?.leave();
  }, [client]);

  if (!client || !call) return <div>Setting up video session...</div>;

  return (
    <StreamVideo client={client}>
      <StreamTheme className="material-minimal">
        <StreamCall call={call}>
          <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '10px', background: '#333', color: '#fff' }}>
              Connected as: <strong>{userId}</strong>
            </div>
            <SpeakerLayout />
            <CallControls onLeave={() => window.location.reload()} />
            <MyParticipantList/>
          </div>
        </StreamCall>
      </StreamTheme>
    </StreamVideo>
  );
};

export default App;