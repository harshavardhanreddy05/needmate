import { useState } from 'react';
import axios from 'axios';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Mic, Search, Sparkles } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Auth({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const response = await axios.post(`${API}${endpoint}`, {
        email,
        password
      });

      onLogin(response.data.user, response.data.access_token);
      toast.success(isLogin ? 'Welcome back!' : 'Account created successfully!');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen gradient-ocean flex items-center justify-center p-4">
      {/* Decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-20 -right-20 w-96 h-96 bg-blue-200 rounded-full opacity-20 blur-3xl"></div>
        <div className="absolute -bottom-20 -left-20 w-96 h-96 bg-cyan-200 rounded-full opacity-20 blur-3xl"></div>
      </div>

      <div className="w-full max-w-6xl relative z-10">
        <div className="grid md:grid-cols-2 gap-8 items-center">
          {/* Left side - Branding */}
          <div className="text-center md:text-left animate-fade-in">
            <div className="inline-flex items-center gap-2 mb-6">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-4xl font-bold text-gray-900">NeedMate</h1>
            </div>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Discover Products with
              <span className="block mt-2 bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                Your Voice & AI
              </span>
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              Simply describe what you need in natural language. Our AI understands your intent and finds the perfect products for you.
            </p>
            <div className="flex flex-wrap gap-4 justify-center md:justify-start">
              <div className="flex items-center gap-2 text-gray-700">
                <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-md">
                  <Mic className="w-5 h-5 text-blue-600" />
                </div>
                <span className="font-medium">Voice Search</span>
              </div>
              <div className="flex items-center gap-2 text-gray-700">
                <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-md">
                  <Search className="w-5 h-5 text-cyan-600" />
                </div>
                <span className="font-medium">Smart Search</span>
              </div>
            </div>
          </div>

          {/* Right side - Auth form */}
          <div className="glass rounded-3xl p-8 shadow-2xl animate-scale-in">
            <div className="mb-6">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                {isLogin ? 'Welcome Back' : 'Get Started'}
              </h3>
              <p className="text-gray-600">
                {isLogin ? 'Sign in to continue shopping' : 'Create your account to start'}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Label htmlFor="email" className="text-gray-700 font-medium">Email</Label>
                <Input
                  id="email"
                  type="email"
                  data-testid="auth-email-input"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="mt-1.5 h-12 bg-white/80 border-gray-200 focus:border-blue-400 focus:ring-blue-400"
                />
              </div>

              <div>
                <Label htmlFor="password" className="text-gray-700 font-medium">Password</Label>
                <Input
                  id="password"
                  type="password"
                  data-testid="auth-password-input"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="mt-1.5 h-12 bg-white/80 border-gray-200 focus:border-blue-400 focus:ring-blue-400"
                />
              </div>

              <Button
                type="submit"
                data-testid="auth-submit-button"
                disabled={loading}
                className="w-full h-12 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold rounded-xl btn-hover"
              >
                {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <button
                type="button"
                data-testid="auth-toggle-button"
                onClick={() => setIsLogin(!isLogin)}
                className="text-blue-600 hover:text-blue-700 font-medium"
              >
                {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}