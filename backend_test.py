import requests
import sys
import json
from datetime import datetime

class NeedMateAPITester:
    def __init__(self, base_url="https://shop-with-voice.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.token = None
        self.user_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        
        if self.token:
            test_headers['Authorization'] = f'Bearer {self.token}'
        
        if headers:
            test_headers.update(headers)

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        print(f"   Method: {method}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=test_headers, timeout=30)

            print(f"   Status: {response.status_code}")
            
            success = response.status_code == expected_status
            response_data = {}
            
            try:
                response_data = response.json()
                if success:
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
            except:
                if not success:
                    print(f"   Response text: {response.text[:200]}...")

            if success:
                self.log_test(name, True)
                return True, response_data
            else:
                self.log_test(name, False, f"Expected {expected_status}, got {response.status_code}")
                return False, response_data

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "/", 200)

    def test_register(self):
        """Test user registration"""
        test_email = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com"
        test_password = "TestPass123!"
        
        success, response = self.run_test(
            "User Registration",
            "POST",
            "/auth/register",
            200,
            data={"email": test_email, "password": test_password}
        )
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            self.user_id = response['user']['id']
            print(f"   Registered user: {test_email}")
            print(f"   User ID: {self.user_id}")
            return True, test_email, test_password
        return False, None, None

    def test_login(self, email, password):
        """Test user login"""
        success, response = self.run_test(
            "User Login",
            "POST",
            "/auth/login",
            200,
            data={"email": email, "password": password}
        )
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            self.user_id = response['user']['id']
            return True
        return False

    def test_get_me(self):
        """Test get current user"""
        success, response = self.run_test(
            "Get Current User",
            "GET",
            "/auth/me",
            200
        )
        return success

    def test_search_products(self):
        """Test product search"""
        search_queries = [
            {"query": "laptop", "query_type": "text"},
            {"query": "wireless headphones", "query_type": "text"},
            {"query": "running shoes", "query_type": "voice"}
        ]
        
        all_success = True
        for i, search_data in enumerate(search_queries):
            success, response = self.run_test(
                f"Product Search {i+1} - '{search_data['query']}'",
                "POST",
                "/search",
                200,
                data=search_data
            )
            
            if success:
                # Validate response structure
                if 'category' in response and 'confidence' in response and 'products' in response:
                    print(f"   Category: {response['category']}")
                    print(f"   Confidence: {response['confidence']:.2f}")
                    print(f"   Products found: {len(response['products'])}")
                    
                    # Check product structure
                    if response['products']:
                        product = response['products'][0]
                        required_fields = ['product_id', 'product_title', 'product_url']
                        missing_fields = [field for field in required_fields if field not in product]
                        if missing_fields:
                            self.log_test(f"Product Structure Validation {i+1}", False, f"Missing fields: {missing_fields}")
                            all_success = False
                        else:
                            self.log_test(f"Product Structure Validation {i+1}", True)
                else:
                    self.log_test(f"Search Response Structure {i+1}", False, "Missing required fields in response")
                    all_success = False
            else:
                all_success = False
        
        return all_success

    def test_search_history(self):
        """Test search history retrieval"""
        success, response = self.run_test(
            "Get Search History",
            "GET",
            "/history",
            200
        )
        
        if success:
            print(f"   History items: {len(response)}")
            if response:
                # Validate history item structure
                item = response[0]
                required_fields = ['id', 'user_id', 'query', 'query_type', 'timestamp']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    self.log_test("History Item Structure", False, f"Missing fields: {missing_fields}")
                    return False
                else:
                    self.log_test("History Item Structure", True)
                    print(f"   Latest search: {item['query']}")
        
        return success

    def test_invalid_auth(self):
        """Test invalid authentication"""
        # Save current token
        original_token = self.token
        
        # Test with invalid token
        self.token = "invalid_token_123"
        success, _ = self.run_test(
            "Invalid Token Test",
            "GET",
            "/auth/me",
            401
        )
        
        # Test with no token
        self.token = None
        success2, _ = self.run_test(
            "No Token Test",
            "GET",
            "/auth/me",
            403  # FastAPI HTTPBearer returns 403 for missing token
        )
        
        # Restore original token
        self.token = original_token
        
        return success and success2

    def test_duplicate_registration(self, email):
        """Test duplicate email registration"""
        success, _ = self.run_test(
            "Duplicate Registration Test",
            "POST",
            "/auth/register",
            400,
            data={"email": email, "password": "AnotherPass123!"}
        )
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting NeedMate API Tests")
        print("=" * 50)
        
        # Test 1: Root endpoint
        self.test_root_endpoint()
        
        # Test 2: User registration
        reg_success, test_email, test_password = self.test_register()
        if not reg_success:
            print("❌ Registration failed, stopping tests")
            return self.get_results()
        
        # Test 3: Get current user
        self.test_get_me()
        
        # Test 4: Product search (multiple queries)
        self.test_search_products()
        
        # Test 5: Search history
        self.test_search_history()
        
        # Test 6: User login with same credentials
        self.test_login(test_email, test_password)
        
        # Test 7: Invalid authentication
        self.test_invalid_auth()
        
        # Test 8: Duplicate registration
        self.test_duplicate_registration(test_email)
        
        return self.get_results()

    def get_results(self):
        """Get test results summary"""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        failed_tests = [test for test in self.test_results if not test['success']]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        return {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": self.tests_passed/self.tests_run*100 if self.tests_run > 0 else 0,
            "test_details": self.test_results
        }

def main():
    tester = NeedMateAPITester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    return 0 if results["failed_tests"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())