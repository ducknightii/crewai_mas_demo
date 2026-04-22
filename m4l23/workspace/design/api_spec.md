# Employee Leave Management System - API Specification

## Base URL
`http://localhost:8000/api`

## Common Data Types

### Leave Record Schema
```json
{
  "id": "string",
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "days": "number",
  "apply_time": "string",
  "status": "string",
  "approver": "string",
  "reason": "string"
}
```

### Leave Type Enum
- `annual_leave` (年假)
- `sick_leave` (病假)
- `personal_leave` (事假)
- `marriage_leave` (婚假)
- `maternity_leave` (产假)
- `paternity_leave` (陪产假)
- `bereavement_leave` (丧假)

### Status Enum
- `pending` (待审批)
- `approved` (已批准)
- `rejected` (已拒绝)

## Endpoints

### GET /leaves
Retrieve all leave records with optional filtering.

**Query Parameters:**
- `employee_name` (string, optional): Filter by employee name
- `leave_type` (string, optional): Filter by leave type (one of the enum values)
- `status` (string, optional): Filter by status (one of the enum values)

**Response (200 OK):**
```json
[
  {
    "id": "string",
    "employee_id": "string",
    "employee_name": "string",
    "leave_type": "string",
    "start_date": "string",
    "end_date": "string",
    "days": "number",
    "apply_time": "string",
    "status": "string",
    "approver": "string",
    "reason": "string"
  }
]
```

**Error Responses:**
- `400 Bad Request`: Invalid query parameters
- `500 Internal Server Error`: Server error

---

### POST /leaves
Create a new leave record.

**Request Body:**
```json
{
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "reason": "string"
}
```

**Response (201 Created):**
```json
{
  "id": "string",
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "days": "number",
  "apply_time": "string",
  "status": "string",
  "approver": "string",
  "reason": "string"
}
```

**Error Responses:**
- `400 Bad Request`: Missing required fields, invalid date format, end_date < start_date, invalid leave_type or status
- `422 Unprocessable Entity`: Validation errors (e.g., reason too long > 200 chars)
- `500 Internal Server Error`: Server error

---

### GET /leaves/{id}
Retrieve a specific leave record by ID.

**Path Parameters:**
- `id` (string): The unique identifier of the leave record

**Response (200 OK):**
```json
{
  "id": "string",
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "days": "number",
  "apply_time": "string",
  "status": "string",
  "approver": "string",
  "reason": "string"
}
```

**Error Responses:**
- `404 Not Found`: Leave record not found
- `500 Internal Server Error`: Server error

---

### PUT /leaves/{id}
Update an existing leave record. Only records with status "pending" can be updated.

**Path Parameters:**
- `id` (string): The unique identifier of the leave record

**Request Body:**
```json
{
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "reason": "string"
}
```

**Response (200 OK):**
```json
{
  "id": "string",
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "days": "number",
  "apply_time": "string",
  "status": "string",
  "approver": "string",
  "reason": "string"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid data, end_date < start_date, invalid leave_type
- `403 Forbidden`: Cannot update record with status other than "pending"
- `404 Not Found`: Leave record not found
- `422 Unprocessable Entity`: Validation errors (e.g., reason too long > 200 chars)
- `500 Internal Server Error`: Server error

---

### DELETE /leaves/{id}
Delete a leave record. Only records with status "pending" can be deleted.

**Path Parameters:**
- `id` (string): The unique identifier of the leave record

**Response (204 No Content):**
No response body

**Error Responses:**
- `403 Forbidden`: Cannot delete record with status other than "pending"
- `404 Not Found`: Leave record not found
- `500 Internal Server Error`: Server error

---

### PATCH /leaves/{id}/status
Update the status of a leave record (for approval/rejection).

**Path Parameters:**
- `id` (string): The unique identifier of the leave record

**Request Body:**
```json
{
  "status": "string",
  "approver": "string"
}
```

**Response (200 OK):**
```json
{
  "id": "string",
  "employee_id": "string",
  "employee_name": "string",
  "leave_type": "string",
  "start_date": "string",
  "end_date": "string",
  "days": "number",
  "apply_time": "string",
  "status": "string",
  "approver": "string",
  "reason": "string"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid status value (must be "approved" or "rejected")
- `403 Forbidden`: Cannot update status of record that is not "pending"
- `404 Not Found`: Leave record not found
- `500 Internal Server Error`: Server error

## Common Error Response Format
All error responses follow this format:
```json
{
  "detail": "string"
}
```

## Date Format
All date fields use ISO 8601 format: `YYYY-MM-DD`

## Datetime Format
All datetime fields use ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.sssZ`